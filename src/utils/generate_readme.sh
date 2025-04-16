git rm ../../README_files/*
jupyter nbconvert --to markdown --output-dir ../../ ../../README.ipynb
sed -i -r 's/\x1B\[[0-9;]*[mK]//g' ../../README.md
git add -f ../../README_files/*.png
git add ../../README.md
