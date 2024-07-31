import urllib3
import argparse

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

    #for file_to_download in files_to_download:
    #    print(file_to_download)
    file_to_download = files_to_download[0]
    url = initial_url+file_to_download
    print(url)
    resp = urllib3.request("GET", initial_url, retries=False, timeout=30)
    print(resp.status)
    print(resp.data)

if __name__ == '__main__':
    main()
