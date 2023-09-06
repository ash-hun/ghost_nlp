# ========================================================================
# 구성 시나리오
# 1. "{Category}-{Folder NAME}-{Year}-{Status}" 형식의 폴더를 구성
#   ex. "(NLP)-(RNN Overview}-{2019)-(Processing)"
# 1-1. 폴더 안에 MarkDown 파일를 만들고 논문리뷰 작성, 파일명은 논문명
#   ex. "Recurrent Neural Networks (RNNs) : A gentle Introduction and Overview" 
# 2. Local 환경에서 1~1.1 단계수행 후 github repository에 push 수행
# 3. 정상적으로 push수행 시 Git action에 의해 CI/CD가 수행
# 4. 정상적으로 Git Action에 의해 CI/CD 수행 시 'update_readme.py'가 수행
# 5. update_readme.py는 새로 푸쉬된 폴더 명과 그 안의 파일명을 추출해 README.md를 업데이트 한다.
#       category = "NLP" -> category
#       date = "23 Nov 2019" -> yeaer
#       title = "Paper Title" -> title
#       status = "processing" -> status
#   "| {category} | {year} | {title} | {status} |"의 형태로 README에 추가됨.

import os

readme_file_path = "/home/runner/work/ghost_nlp/ghost_nlp/README.md"
base_dir = r'/home/runner/work/ghost_nlp/ghost_nlp/'
# readme_file_path = "../README.md" # TEST Code
# base_dir = "../" # TEST Code

each_dir_to_time = []
for each_dir_name in os.listdir(base_dir):
    if each_dir_name == 'script':
        continue
    if '.' not in each_dir_name:
        each_dir_to_time.append([each_dir_name, os.path.getctime(base_dir+each_dir_name)])

most_recent_dir = max(each_dir_to_time, key=lambda x: x[1])[0]
tmp = list(list(most_recent_dir.split('-')))
print(tmp)
category, year, folderName, status = tmp[0], tmp[1], tmp[2], tmp[3]
# print(tmp)
# print(category, folderName, year, status)
curr_folder_path = f"/home/runner/work/ghost_nlp/ghost_nlp/{most_recent_dir}/"
# curr_folder_path = f"../{most_recent_dir}" # Testing Code

target_paper_review_file = ""
checkList = []
for item in os.listdir(curr_folder_path):
    checkList.append(item)
    if ".md" in item or ".markdown" in item:
        target_paper_review_file += item
title = target_paper_review_file.split('.')[0]

res = f"| {category} | {year} | {title} | {status} |"

# Update README.md Contents
readme = open(readme_file_path, "a+")
readme.write(res+'\n')

readme.close()
