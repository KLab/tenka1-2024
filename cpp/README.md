```shell
# WSL2 Ubuntu 22.04.2 LTS
# g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
sudo apt install libcurl4-openssl-dev nlohmann-json3-dev
g++ -std=c++17 -O2 main.cpp -lcurl -o main
./main
```