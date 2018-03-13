#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"
code-variants --run --copies 100 ./play.py __results/ || exit 1

RESULTS=$(grep -r 'Game result:' __results) || exit 1
RESULTS=$(sed -r -e 's|^.*/([a-z]+)_([a-z]+)_.*\s([^\s]+)$|\1 \2 \3|' <<<"$RESULTS") || exit 1

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

printf "%12s" ""
for p2 in $PLAYERS; do
    printf "%12s" "W: $p2"
done
echo

for p1 in $PLAYERS; do
    printf "%12s" "B: $p1"
    for p2 in $PLAYERS; do
        printf "%12s" "${COUNT["${p1}_${p2}_X"]:-0}/${COUNT["${p1}_${p2}_draw"]:-0}/${COUNT["${p1}_${p2}_O"]:-0}"
    done
    echo
done
