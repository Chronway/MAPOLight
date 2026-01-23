REM python .\box_and_line2.py -a ppo uniform maxpressure websters --save ppo_wait.pdf
REM python .\box_and_line2.py -a a3c uniform maxpressure websters --save a3c_wait.pdf
REM python .\box_and_line2.py -a sac uniform maxpressure websters --save sac_wait.pdf
@REM python .\box_and_line2.py -a ppo uniform maxpressure websters -t total_stopped --save ppo_qlen.pdf -l "upper left"
@REM python .\box_and_line2.py -a a3c uniform maxpressure websters -t total_stopped --save a3c_qlen.pdf -l "upper left"
@REM python .\box_and_line2.py -a sac uniform maxpressure websters -t total_stopped --save sac_qlen.pdf -l "upper left"
@REM python .\box_and_line2.py -a ppo uniform maxpressure websters -t avg_speed --save ppo_spee.pdf
@REM python .\box_and_line2.py -a a3c uniform maxpressure websters -t avg_speed --save a3c_spee.pdf
@REM python .\box_and_line2.py -a sac uniform maxpressure websters -t avg_speed --save sac_spee.pdf
@REM python .\box_and_line2.py -p 1.0 uniform maxpressure websters -a ppo a3c sac uniform maxpressure websters --save fo_wait.pdf
python .\box_and_line2.py -p 1.0 uniform maxpressure websters -a ppo a3c sac uniform maxpressure websters -t total_stopped --save "4x4 fo_qlen.pdf" -l "upper left"
python .\box_and_line2.py -p 1.0 uniform maxpressure websters -a ppo a3c sac uniform maxpressure websters -t avg_speed --save "4x4 fo_spee.pdf"

@REM python .\lineplot.py --save "4x4 Corr.pdf"