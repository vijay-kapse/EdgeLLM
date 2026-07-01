| Backend | Precision | Device | Size (MB) | Latency (s) | Throughput (tok/s) | Peak RAM (MB) | Perplexity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pytorch | fp32 | mps | 942.32 | 2.9633 ± 0.0175 | 21.6 | 2495.7 | 19.0 |
| ort-cpu | fp32 | cpu | 2404.97 | 3.2494 ± 0.0898 | 19.71 | 2910.8 | 19.0 |
| ort-cpu-int8 | int8 | cpu | 604.98 | 1.6732 ± 0.0401 | 38.27 | 2861.8 | 20.097 |
| ort-cpu-int4 | int4 | cpu | 770.93 | 3.2065 ± 0.0703 | 19.97 | 3194.2 | 24.575 |
| pytorch | int8 | cpu | — | 4.0221 ± 0.0777 | 15.92 | 5396.5 | 58.423 |
