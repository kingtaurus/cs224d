ASSIGNMENT=assignment1
wget http://cs224d.stanford.edu/${ASSIGNMENT}/${ASSIGNMENT}.zip
rm -f ${ASSIGNMENT}.pdf
wget http://cs224d.stanford.edu/${ASSIGNMENT}/${ASSIGNMENT}.pdf
unzip ${ASSIGNMENT}.zip
rm -f ${ASSIGNMENT}.zip
wget https://raw.githubusercontent.com/qipeng/nbutils/master/updateAssignment.py

echo Update in progress...
python updateAssignment.py . ${ASSIGNMENT}

rm -rf updateAssignment.py
rm -rf ${ASSIGNMENT}
rm -rf __MACOSX
echo Done!