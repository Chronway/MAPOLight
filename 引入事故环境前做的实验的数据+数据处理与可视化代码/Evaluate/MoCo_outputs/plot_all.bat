REM python .\box_and_line2.py --save wait_no_a3c.png
python .\box_and_line2.py -t total_stopped --save "Moco qlen2.pdf"
python .\box_and_line2.py -t avg_speed --save "Moco spee2.pdf"

python .\lineplot.py --save "MoCo Corr.pdf"