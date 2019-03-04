x=$(find /media/sdf/NIR-Database/images/ -type f | shuf -n 200000)

for file in $x; do
    cp $file /media/sdf/NIR-Database/images-rand
done

