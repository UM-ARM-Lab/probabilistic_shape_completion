export fileid=19Ju4ZCBu2-B75Vq-NZ6EpDUTb4djeLi2
export filename=train_mod/model.cptk.data-00000-of-00001

wget --save-cookies cookies.txt 'https://drive.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O $filename \
     'https://drive.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)

rm confirm.txt
rm cookies.txt
