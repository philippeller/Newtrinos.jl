git rm ../../README_files/*
jupyter nbconvert --to markdown --output-dir ../../ ../../README.ipynb
git add -f ../../README_files/*.png
git add ../../README.md
