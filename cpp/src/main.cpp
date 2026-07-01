// EdgeLLM C++ inference harness.
//
// Loads a quantized (or FP32) ONNX causal-LM exported by Optimum and runs
// autoregressive greedy decoding *with a KV cache* through the ONNX Runtime
// C++ API, then prints latency and tokens/sec. Tokenization and detokenization
// live on the Python side (see `edgellm encode` / `edgellm decode`); this binary
// owns the actual model inference, which is the point of the exercise.
//
// Usage:
//   edgellm_infer --model <dir> --tokens "1 2 3 ..." [--max-new-tokens 64] [--threads N]
//
// The model directory must contain model.onnx and config.json. Architecture
// parameters (layers, KV heads, head dim) are read from config.json and can be
// overridden on the command line.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string model_dir;
  std::vector<int64_t> tokens;
  int max_new_tokens = 64;
  int threads = 0;  // 0 = ORT default
  int num_layers = -1;
  int kv_heads = -1;
  int head_dim = -1;
};

[[noreturn]] void die(const std::string& msg) {
  std::cerr << "error: " << msg << "\n";
  std::exit(1);
}

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) die("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Minimal extraction of an integer field from a JSON blob: finds "key" then the
// next integer after the following ':'. Sufficient for the flat config.json
// fields we need; avoids pulling in a JSON dependency.
bool json_int(const std::string& json, const std::string& key, long& out) {
  const std::string needle = "\"" + key + "\"";
  size_t pos = json.find(needle);
  if (pos == std::string::npos) return false;
  pos = json.find(':', pos);
  if (pos == std::string::npos) return false;
  ++pos;
  while (pos < json.size() && !std::isdigit(json[pos]) && json[pos] != '-') ++pos;
  if (pos >= json.size()) return false;
  out = std::strtol(json.c_str() + pos, nullptr, 10);
  return true;
}

std::vector<int64_t> parse_tokens(const std::string& s) {
  std::vector<int64_t> out;
  std::string cur;
  for (char c : s) {
    if (c == ',' || c == ' ' || c == '\t' || c == '\n') {
      if (!cur.empty()) { out.push_back(std::stoll(cur)); cur.clear(); }
    } else {
      cur += c;
    }
  }
  if (!cur.empty()) out.push_back(std::stoll(cur));
  return out;
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) die("missing value for " + k);
      return argv[++i];
    };
    if (k == "--model") a.model_dir = next();
    else if (k == "--tokens") a.tokens = parse_tokens(next());
    else if (k == "--max-new-tokens") a.max_new_tokens = std::stoi(next());
    else if (k == "--threads") a.threads = std::stoi(next());
    else if (k == "--layers") a.num_layers = std::stoi(next());
    else if (k == "--kv-heads") a.kv_heads = std::stoi(next());
    else if (k == "--head-dim") a.head_dim = std::stoi(next());
    else die("unknown argument: " + k);
  }
  if (a.model_dir.empty()) die("--model is required");
  if (a.tokens.empty()) die("--tokens is required (space/comma separated int64 ids)");
  return a;
}

// argmax over the last position's vocab row of a [1, seq, vocab] logits tensor.
int64_t argmax_last(const float* logits, int64_t seq, int64_t vocab) {
  const float* row = logits + (seq - 1) * vocab;
  int64_t best = 0;
  float best_val = row[0];
  for (int64_t v = 1; v < vocab; ++v) {
    if (row[v] > best_val) { best_val = row[v]; best = v; }
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  // --- Read architecture from config.json (CLI overrides win) ---
  const std::string cfg = read_file(args.model_dir + "/config.json");
  long v = 0;
  int num_layers = args.num_layers;
  int kv_heads = args.kv_heads;
  int head_dim = args.head_dim;
  if (num_layers < 0 && json_int(cfg, "num_hidden_layers", v)) num_layers = static_cast<int>(v);
  if (kv_heads < 0 && json_int(cfg, "num_key_value_heads", v)) kv_heads = static_cast<int>(v);
  if (head_dim < 0) {
    long hidden = 0, heads = 0;
    if (json_int(cfg, "hidden_size", hidden) && json_int(cfg, "num_attention_heads", heads) &&
        heads > 0) {
      head_dim = static_cast<int>(hidden / heads);
    }
  }
  if (num_layers <= 0 || kv_heads <= 0 || head_dim <= 0)
    die("could not determine model architecture; pass --layers/--kv-heads/--head-dim");

  std::cerr << "model: " << args.model_dir << "  layers=" << num_layers
            << " kv_heads=" << kv_heads << " head_dim=" << head_dim << "\n";

  // --- Build ONNX Runtime session ---
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edgellm");
  Ort::SessionOptions opts;
  if (args.threads > 0) opts.SetIntraOpNumThreads(args.threads);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  const std::string model_path = args.model_dir + "/model.onnx";
  Ort::Session session(env, model_path.c_str(), opts);

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // --- Assemble static input/output name lists ---
  std::vector<std::string> in_names_s = {"input_ids", "attention_mask", "position_ids"};
  std::vector<std::string> out_names_s = {"logits"};
  for (int l = 0; l < num_layers; ++l) {
    in_names_s.push_back("past_key_values." + std::to_string(l) + ".key");
    in_names_s.push_back("past_key_values." + std::to_string(l) + ".value");
    out_names_s.push_back("present." + std::to_string(l) + ".key");
    out_names_s.push_back("present." + std::to_string(l) + ".value");
  }
  std::vector<const char*> in_names, out_names;
  for (auto& s : in_names_s) in_names.push_back(s.c_str());
  for (auto& s : out_names_s) out_names.push_back(s.c_str());

  // --- Initialise KV cache with empty (zero-length) past tensors ---
  const int n_kv = num_layers * 2;
  float dummy = 0.0f;
  std::vector<Ort::Value> past;
  past.reserve(n_kv);
  for (int i = 0; i < n_kv; ++i) {
    std::array<int64_t, 4> shape = {1, kv_heads, 0, head_dim};
    past.push_back(Ort::Value::CreateTensor<float>(mem, &dummy, 0, shape.data(), shape.size()));
  }

  std::vector<int64_t> generated;
  generated.reserve(args.max_new_tokens);

  auto run_step = [&](const std::vector<int64_t>& ids, int64_t past_len) -> int64_t {
    const int64_t cur = static_cast<int64_t>(ids.size());
    const int64_t total = past_len + cur;

    // Backing buffers must outlive the Run() call.
    std::vector<int64_t> input_ids = ids;
    std::vector<int64_t> attn(total, 1);
    std::vector<int64_t> pos(cur);
    for (int64_t j = 0; j < cur; ++j) pos[j] = past_len + j;

    std::array<int64_t, 2> ids_shape = {1, cur};
    std::array<int64_t, 2> attn_shape = {1, total};
    std::array<int64_t, 2> pos_shape = {1, cur};

    std::vector<Ort::Value> inputs;
    inputs.reserve(3 + n_kv);
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, input_ids.data(), input_ids.size(),
                                                       ids_shape.data(), ids_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, attn.data(), attn.size(),
                                                       attn_shape.data(), attn_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, pos.data(), pos.size(),
                                                       pos_shape.data(), pos_shape.size()));
    for (auto& p : past) inputs.push_back(std::move(p));

    auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(),
                               inputs.size(), out_names.data(), out_names.size());

    // logits -> argmax of the final position.
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();  // [1, seq, vocab]
    const int64_t seq = shape[1];
    const int64_t vocab = shape[2];
    const int64_t next = argmax_last(outputs[0].GetTensorData<float>(), seq, vocab);

    // present -> next past.
    past.clear();
    for (int i = 0; i < n_kv; ++i) past.push_back(std::move(outputs[1 + i]));
    return next;
  };

  using clock = std::chrono::steady_clock;

  // --- Prefill on the full prompt ---
  auto t0 = clock::now();
  int64_t next = run_step(args.tokens, 0);
  auto t1 = clock::now();
  generated.push_back(next);
  int64_t past_len = static_cast<int64_t>(args.tokens.size());

  // --- Decode loop (greedy, one token at a time) ---
  for (int step = 1; step < args.max_new_tokens; ++step) {
    next = run_step({next}, past_len);
    ++past_len;
    generated.push_back(next);
  }
  auto t2 = clock::now();

  const double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double decode_s = std::chrono::duration<double>(t2 - t1).count();
  const double total_s = std::chrono::duration<double>(t2 - t0).count();
  const int decode_tokens = args.max_new_tokens - 1;

  std::cout << "GENERATED_IDS:";
  for (int64_t id : generated) std::cout << " " << id;
  std::cout << "\n";
  std::cout << "prompt_tokens: " << args.tokens.size() << "\n";
  std::cout << "generated_tokens: " << generated.size() << "\n";
  std::cout << "prefill_ms: " << prefill_ms << "\n";
  std::cout << "decode_tokens_per_s: "
            << (decode_s > 0 ? decode_tokens / decode_s : 0.0) << "\n";
  std::cout << "total_s: " << total_s << "\n";
  std::cout << "overall_tokens_per_s: "
            << (total_s > 0 ? generated.size() / total_s : 0.0) << "\n";
  return 0;
}
