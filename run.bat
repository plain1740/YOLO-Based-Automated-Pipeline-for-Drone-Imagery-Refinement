
call conda activate server_env

python src/0_process_video.py

python src/1_deduplicat_image.py

python src/2_past_audit.py

python src/2_pre_audit.py

pause