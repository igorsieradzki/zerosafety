#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"
code-variants --run --copies 100 ./play.py __results/ || exit 1

RESULTS=$(grep -r 'Game result:' __results) || exit 1
RESULTS=$(sed -r -e 's|^.*/([a-z]+)_([a-z]+)_.*\s([^\s]+)$|\1 \2 \3|' <<<"$RESULTS") || exit 1

declare -A WIN
declare -A LOSS
declare -A DRAW
declare -A PLAYER

while read p1 p2 result; do
    PLAYER["$p1"]=1
    PLAYER["$p2"]=1
    case "x$result" in
        xX)
            COUNT=${WIN["${p1}_${p2}"]:-0}
            let COUNT=COUNT+1
            WIN["${p1}_${p2}"]="$COUNT"
        ;;
        xO)
            COUNT=${LOSS["${p1}_${p2}"]:-0}
            let COUNT=COUNT+1
            LOSS["${p1}_${p2}"]="$COUNT"
        ;;
        xdraw)
            COUNT=${DRAW["${p1}_${p2}"]:-0}
            let COUNT=COUNT+1
            DRAW["${p1}_${p2}"]="$COUNT"
        ;;
        *)
            echo "Unrecognized result: $result" 1>&2
            exit 1
        ;;
    esac
done <<<"$RESULTS"

printf "%12s" ""
for p2 in "${!PLAYER[@]}"; do
    printf "%12s" "W: $p2"
done
echo

for p1 in "${!PLAYER[@]}"; do
    printf "%12s" "B: $p1"
    for p2 in "${!PLAYER[@]}"; do
        printf "%12s" "${WIN["${p1}_${p2}"]:-0}/${DRAW["${p1}_${p2}"]:-0}/${LOSS["${p1}_${p2}"]:-0}"
    done
    echo
done
