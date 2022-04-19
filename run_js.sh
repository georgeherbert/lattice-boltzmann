#!/bin/bash
for i in {1..112}
do
    sbatch js_procs/js_$i
done