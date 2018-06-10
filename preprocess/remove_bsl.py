import os

# go through all the ones on the WDRed8Tb1

disk_one = "/run/media/jacob/WDRed8Tb1/"

# Go through all the ones on the WDRed8Tb2

disk_two = "/run/media/jacob/WDRed8Tb2/ihp-pc41.ethz.ch/public/phs/"

# And ones on Seagate

disk_three = "/run/media/jacob/Seagate/ihp-pc41.ethz.ch/public/phs/"
# And ones on HDD

disk_four = "/run/media/jacob/HDD/2014/"

disks = [disk_four, disk_three, disk_two, disk_one]

for disk in disks:
    for subdir, dirs, files in os.walk(disk):
        for file in files:
            if ".bsl.jsonl.gz" in file:
                path = os.path.join(subdir, file)
                print(path)
                os.remove(path)
