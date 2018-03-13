#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"
code-variants --run --copies 100 ./play.py __results/ || exit 1

RESULTS=$(grep -r 'Game result:' __results) || exit 1
RESULTS=$(sed -r -e 's|^.*/([a-z]+)_([a-z]+)_.*\s([^\s]+)$|\1 \2 \3|' \
    <<<"$RESULTS") || exit 1

declare -A COUNT

while read p1 p2 result; do
    case "x$result" in
        xX) ;;
        xO) ;;
        xdraw) ;;
        *)
            echo "Unrecognized result: $result" 1>&2
            exit 1
        ;;
    esac

    C=${COUNT["${p1}_${p2}_${result}"]:-0}
    let C=C+1
    COUNT["${p1}_${p2}_${result}"]="$C"
done <<<"$RESULTS"

PLAYERS=$(while read p1 p2 result; do \
    echo "$p1"; echo "$p2"; done <<<"$RESULTS" | sort -u)

WIDTH=15

printf "%${WIDTH}s " ""
for p2 in $PLAYERS; do
    P="W: $p2"
    printf "%${WIDTH}s " "${P::${WIDTH}}"
done
echo

for p1 in $PLAYERS; do
    P="B: $p1"
    printf "%${WIDTH}s " "${P::${WIDTH}}"
    for p2 in $PLAYERS; do
        P="${COUNT["${p1}_${p2}_X"]:-0}"
        P+="/${COUNT["${p1}_${p2}_draw"]:-0}"
        P+="/${COUNT["${p1}_${p2}_O"]:-0}"
        printf "%${WIDTH}s " "${P::${WIDTH}}"
    done
    echo
done
