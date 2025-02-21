#!/bin/bash

#################################################################################################
# frameの画像ファイルすべての番号部分を、連番になるように振り直す
# $ ./rename.sh [path]
#
# 連番に振りなおしたのち、ffmpegでgif等に変換する。
# $ ffmpeg -i frame_%03d.png -vf palettegen palette.png
# $ ffmpeg -framerate 5 -i frame_%03d.png -i palette.png -filter_complex "paletteuse" output.gif
##################################################################################################

counter=0
frame_path=$1

echo $frame_path

for file in $frame_path/progress*.png; do
  new_filename=$(printf $frame_path"/progress%03d.png" "$counter")

  mv "$file" "$new_filename"

  counter=$((counter + 1))
done
# counter=1
# frame_path=$1

# echo $frame_path
# echo "file path": $frame_path/frame_1.png

# for file in $frame_path/frame_*.png; do
#   new_filename=$(printf $frame_path"/frame_%03d.png" "$counter")

#   mv "$file" "$new_filename"

#   counter=$((counter + 1))
# done