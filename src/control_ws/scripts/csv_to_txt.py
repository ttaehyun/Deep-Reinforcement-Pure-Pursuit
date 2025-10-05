import csv

# 변환할 CSV 파일 이름과 새로 만들 TXT 파일 이름을 지정합니다.
csv_file_path = '/home/avl/Downloads/competition_map_testday1_centerline.csv'  # 원본 CSV 파일 경로
txt_file_path = '/home/avl/KSAE_FSD_2025/src/control_ws/path/competition_map_testday1_centerlineput.txt' # 저장할 TXT 파일 경로

try:
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            # CSV 파일을 읽기 위한 reader 객체를 생성합니다.
            csv_reader = csv.reader(csv_file)
            
            # CSV 파일의 각 행(row)을 반복하면서 처리합니다.
            for row in csv_reader:
                # 각 행의 데이터(리스트)를 쉼표(,)로 연결하여 하나의 문자열로 만듭니다.
                line = ','.join(row)
                
                # 변환된 문자열을 TXT 파일에 쓰고, 줄바꿈 문자를 추가합니다.
                txt_file.write(line + '\n')

    print(f"'{csv_file_path}' 파일이 '{txt_file_path}' 파일로 성공적으로 변환되었습니다.")

except FileNotFoundError:
    print(f"오류: '{csv_file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")