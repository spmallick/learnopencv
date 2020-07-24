#!/bin/bash

rm -rf logs
mkdir -p logs

sample_dirs=$(find -mindepth 1 -maxdepth 1 -type d)
for s in $sample_dirs
do
  sample=$(basename $s)
  cd $sample

  if [ -e ./.ci_ignore ]; then
      echo -e "$sample: SKIPPED"
      cd ..
      continue
  fi

  if ls *.cpp > /dev/null 2>&1; then
    ../CI/build_one.sh > ../logs/$sample.log 2>&1
    if [ $? -ne 0 ]; then
        echo -e "$sample: \e[31mFAILED\e[0m"
    else
        echo -e "$sample: \e[32mPASSED\e[0m"
    fi
  else
      echo -e "$sample: SKIPPED"
  fi
  cd ..
done
