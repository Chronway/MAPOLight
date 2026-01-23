REM python .\box_and_line2.py --save wait.pdf
python .\box_and_line2.py -t total_stopped --save "5x5 qlen.pdf" -l "upper left"
python .\box_and_line2.py -t avg_speed --save "5x5 spee.pdf"

python .\lineplot.py --save "5x5 Corr.pdf"