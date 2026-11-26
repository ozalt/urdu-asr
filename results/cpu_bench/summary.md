Hardware: AMD Ryzen 5 7600 6-Core Processor (6 cores, 12 threads), 14.7 GB RAM, Linux Mint 22.1. CPU only, 4 threads, llama.cpp b12675. Limited to 4 CPUs [0, 1, 2, 3] and 8.0 GB RAM.
20 FLEURS test clips never seen in training, 205 s of audio. Real-time factor = processing time / audio length (below 1 = faster than real time).

| Run | WER | CER | Real-time factor |
| --- | --- | --- | --- |
| Base, auto language | 100.2% | 85.1% | 0.36 |
| Base, told Urdu | 74.9% | 55.8% | 0.30 |
| Fine-tuned Q8_0, Urdu | 22.1% | 9.3% | 0.28 |
| Fine-tuned Q8_0, Roman | 30.6% | 12.0% | 0.21 |
| Fine-tuned Q4_K_M, Urdu | 23.5% | 9.9% | 0.21 |
| Fine-tuned Q4_K_M, Roman | 32.3% | 13.8% | 0.17 |

| Model | Files (MB) | Peak RAM (MB) | Load (s) |
| --- | --- | --- | --- |
| base | 1184 | 2492 | 1.0 |
| q8_0 | 1018 | 2261 | 1.0 |
| q4_k_m | 776 | 2232 | 1.0 |
