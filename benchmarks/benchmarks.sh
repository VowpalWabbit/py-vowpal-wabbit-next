#!/usr/bin/env bash
set -e

if [ ! -f rcv1.tar.gz ]
then
    echo "Fetching data..."
    wget http://hunch.net/~jl/rcv1.tar.gz
    tar -xvf rcv1.tar.gz
    head -50000 rcv1.train.txt > rcv1.50k.txt
    head -5000 rcv1.train.txt > rcv1.5k.txt
    head -10000 rcv1.train.txt > rcv1.10k.txt
    head -15000 rcv1.train.txt > rcv1.15k.txt
    head -20000 rcv1.train.txt > rcv1.20k.txt
fi

VW_BIN="../out/build/default/ext_libs/vowpal_wabbit/vowpalwabbit/cli/vw"
hyperfine --warmup 5 --export-csv bench1.csv "$VW_BIN --quiet --no_stdin" "python ./cli_runner.py --quiet --no_stdin"
hyperfine --warmup 5 --export-csv bench2.csv "$VW_BIN -d rcv1.5k.txt -q:: -k" "python ./cli_runner.py -d rcv1.5k.txt -q:: -k"
hyperfine --warmup 5 --export-csv bench3.csv "$VW_BIN -d rcv1.5k.txt --quiet -q:: -k -c --passes 5" "python ./cli_runner.py -d rcv1.5k.txt --quiet -q:: -k -c --passes 5"
hyperfine --warmup 5 --export-csv bench4.csv "$VW_BIN -d rcv1.5k.txt --quiet -q:: -k" "python ./cli_runner.py -d rcv1.5k.txt --quiet -q:: -k"
hyperfine --warmup 5 --export-csv bench5.csv "$VW_BIN -d rcv1.10k.txt --quiet -q:: -k" "python ./cli_runner.py -d rcv1.10k.txt --quiet -q:: -k"
hyperfine --warmup 5 --export-csv bench6.csv "$VW_BIN -d rcv1.15k.txt --quiet -q:: -k" "python ./cli_runner.py -d rcv1.15k.txt --quiet -q:: -k"
hyperfine --warmup 5 --export-csv bench7.csv "$VW_BIN -d rcv1.20k.txt --quiet -q:: -k" "python ./cli_runner.py -d rcv1.20k.txt --quiet -q:: -k"
hyperfine --warmup 5 --export-csv bench8.csv "python cache_create.py" "$VW_BIN -d rcv1.10k.txt -c -k --quiet --noop"
