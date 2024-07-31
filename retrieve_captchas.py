import urllib3
import argparse
import os
import time
import random
def request_file(initial_url, file_to_download, output):
    url = initial_url + file_to_download
    resp = urllib3.request("GET", url, retries=False, timeout=30)
    time.sleep(random.randint(1,3))
    if resp.status == 200:
        with open(output+file_to_download, 'wb') as f:
            f.write(resp.data)
        print("Retrieved "+file_to_download)
    else:
        print("Failed to retrieve " + file_to_download + ", retrying")
        request_file(initial_url, file_to_download, output)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list-path', help='File path where the list of files to download is stored', type=str)
    parser.add_argument('--shortname', help='TCD shortname to download files with (eg. cgregg)', type=str)
    parser.add_argument('--output', help='Where to store downloaded files', type=str)
    args = parser.parse_args()

    if args.list_path is None:
        print("Please specify where the list of files to download is stored")
        exit(1)

    if args.shortname is None:
        print("Please specify the TCD shortname to download files with")
        exit(1)

    if args.output is None:
        print("Please specify the output directory")
        exit(1)

    files_to_download = []
    with open(args.list_path, 'r') as list_file:
        for line in list_file:
            files_to_download.append(line.strip())
    initial_url = "https://cs7ns1.scss.tcd.ie/?shortname="+args.shortname+"&myfilename="
    
    files_already_downloaded = os.listdir(args.output)
    print("These files have already been downloaded, so will not be downloaded again")
    print(files_already_downloaded)

    for file_to_download in files_to_download:
        if (file_to_download not in files_already_downloaded):
            request_file(initial_url, file_to_download, args.output)

if __name__ == '__main__':
    main()
