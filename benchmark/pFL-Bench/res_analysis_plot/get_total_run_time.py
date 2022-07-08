import wandb

api = wandb.Api()

name_project = ["daoyuan/pFL-bench", "daoyuan/pfl-bench-best-repeat"]

total_run_time = 0
run_cnt = 0
run_finish_cnt = 0
total_run_time_finished = 0


def convert(seconds):
    seconds_in_day = 60 * 60 * 24
    seconds_in_hour = 60 * 60
    seconds_in_minute = 60

    days = seconds // seconds_in_day
    hours = (seconds - (days * seconds_in_day)) // seconds_in_hour
    minutes = (seconds - (days * seconds_in_day) -
               (hours * seconds_in_hour)) // seconds_in_minute

    return "%d:%02d:%02d" % (days, hours, minutes)


def print_run_time():
    print(f"Total_run_t: {convert(total_run_time)}, run_cnt={run_cnt}")
    print(f"Total_run_t_finished: {convert(total_run_time_finished)}, "
          f"run_cnt_finish={run_finish_cnt}")


for p in name_project:
    runs = api.runs(p)
    for run in runs:
        try:
            if '_runtime' in run.summary:
                time_run = run.summary["_runtime"]
            else:
                time_run = run.summary['_wandb']["runtime"]

            if run.state == "finished":
                total_run_time_finished += time_run
                run_finish_cnt += 1
            total_run_time += time_run
            run_cnt += 1
            if run_cnt % 200 == 0:
                print_run_time()
        except:
            # print("something wrong")
            continue

print_run_time()
