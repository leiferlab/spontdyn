echo ""
echo ""
echo "MAKE SURE YOU ADJUST THE PATHS"
echo "AND REMOVE --export-data FROM LINE 8 OF THIS SCRIPT"
echo ""
echo ""

python raw_spectrum.py --no-normalize --export-data
python plot_datasets_main.py
python plot_datasets_supp.py

cd /projects/LEIFER/francesco/spontdyn/
tar czvf exported_data.tar.gz exported_data/
tar czvf exported_data2.tar.gz exported_data2/ 
