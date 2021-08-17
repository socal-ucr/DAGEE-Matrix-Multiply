#!/bin/bash

if [ $# -eq 0 ]
then
  echo "$0 [--gpu | --cpu | --gpu-dagee | --cpu-dagee]"
  exit
fi

cmd=""

if [ $1 = "--cpu" ]
then
  cmd="CPU_Strassen"
elif [ $1 = "--gpu" ]
then
  cmd="GPU_Strassen"
elif [ $1 = "--gpu-dagee" ]
then
  cmd="GPU_Strassen_dagee"
elif [ $1 = "--cpu-dagee" ]
then
  cmd="CPU_Strassen_dagee"
else
  echo "$0 [--gpu | --cpu | --gpu-dagee | --cpu-dagee]"
  exit
fi


for ((i=16; i<=16384; i=i*2))
do
  echo "Running $cmd $i. Matrix size = $(( i*i ))"
  ./$cmd $i
done
