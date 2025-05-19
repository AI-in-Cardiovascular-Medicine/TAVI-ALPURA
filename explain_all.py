import os
import time

# scripts
events = [
    'cdeath',
    'death'
]

for event in events:
    print(f"\n -------- {event} --------")
    start = time.time()
    os.system(f"Rscript -e \"rmarkdown::render('xai_global.rmd', params=list(event='{event}'))\"")  # global
    print(f"Elapsed time: {time.time() - start}")
