//#include "stdafx.h"
#include "HtpClient.h"

/*time_t now = time(0);
tm* ltm = localtime(&now);
char temp[64];
strftime(temp,sizeof(temp),"%Y-%m-%d",ltm);*/
HtpClient::HtpClient()
{
}

//bool HtpClient::Doget()
//{
//	HINTERNET hInternet;
//	hInternet = InternetOpen(NULL, INTERNET_OPEN_TYPE_PRECONFIG, NULL, NULL, 0);
//	if (hInternet == NULL)
//	{
//		InternetCloseHandle(hInternet);
//		return false;
//	}
//
//	HINTERNET hConnect;
//	INTERNET_PORT nport = 9000;
//	hConnect = InternetConnect(hInternet, _T("127.0.0.1"), nport, NULL, NULL, INTERNET_SERVICE_HTTP, 0, 0);
//	if (hConnect == NULL)
//	{
//		InternetCloseHandle(hConnect);
//		InternetCloseHandle(hInternet);
//		return false;
//	}
//
//	HINTERNET httpFile;
//	httpFile = HttpOpenRequest(hConnect, _T("GET"), _T("blog/index/"), _T("HTTP/1.1"),
//		NULL, 0, INTERNET_FLAG_NO_UI, 1);
//	if (httpFile == NULL)
//	{
//		InternetCloseHandle(httpFile);
//		InternetCloseHandle(hConnect);
//		InternetCloseHandle(hInternet);
//		return false;
//	}
//
//	//LPCTSTR  lpszHeaders = _T("Accept-Charset：utf-8\r\n");
//	//HttpAddRequestHeaders(httpFile, lpszHeaders,24,HTTP_ADDREQ_FLAG_ADD_IF_NEW);
//	HttpSendRequest(httpFile, NULL, 0, NULL, 0);
//
//	FILE* fp;
//	fopen_s(&fp, "C:\\Users\\18521\\Documents\\All_Project\\python\\file.txt", "w");
//	TCHAR* buf = new TCHAR[256]();
//	DWORD buf_read = 0;
//	DWORD buf_len = 256;
//
//	while (1)
//	{
//		if (!InternetReadFile(httpFile, buf, buf_len, &buf_read)) break;
//		if (fwrite(buf, 1, buf_read, fp) != 0) break;
//	}
//
//	delete[] buf;
//	fclose(fp);
//	InternetCloseHandle(httpFile);
//	InternetCloseHandle(hConnect);
//	InternetCloseHandle(hInternet);
//	return true;
//}


//使用libcurl完成http post
int HtpClient::Dopost()
{
	CURLcode rescode = curl_global_init(CURL_GLOBAL_ALL);//建立程序环境
	
	CURL* curl = curl_easy_init();//得到curl_easy句柄
		
	struct curl_slist *headers = NULL;//请求头的容器
	//FILE* fp;//接受回复的文件
	CURLcode res;//状态码
	struct curl_httppost *formpost = NULL;//表单容器
	struct curl_httppost *lastptr = NULL;
			

			
			
			
	/*char szJsonData[256];
	memset(szJsonData, 0, sizeof(szJsonData));
	std::string strJson = "{";
	strJson += "\"level\":\"two\",";
	strJson += "\"description\":\"persons\",";
	strJson += "\"device_name\":\"east04\"";
	strJson += "}";
	strcpy_s(szJsonData,strJson.c_str());*/
	/*curl_formadd(&formpost, &lastptr,
	CURLFORM_COPYNAME, "json",
	CURLFORM_COPYCONTENTS, szJsonData,
	CURLFORM_CONTENTTYPE, "application/json",
	CURLFORM_END);*/
	//if (fopen_s(&fp, "C:\\Users\\18521\\Documents\\All_Project\\python\\file.txt", "w"))  // 返回结果用文件存储
	//	return false;

	//添加请求头
	headers = curl_slist_append(headers, level);
	headers = curl_slist_append(headers, description);
	headers = curl_slist_append(headers, device);
	//headers = curl_slist_append(headers, group);
	//headers = curl_slist_append(headers, place);
	headers = curl_slist_append(headers, datetime);

	//添加表单
	curl_formadd(&formpost, &lastptr,
		CURLFORM_COPYNAME, "photo_alarm",
		CURLFORM_FILE, alarmPicture,
		CURLFORM_FILENAME, alarmPictureName,
		CURLFORM_CONTENTTYPE, "image/jpeg",
		CURLFORM_END);
	/*curl_formadd(&formpost, &lastptr,
		CURLFORM_COPYNAME, "photo_noalarm",
		CURLFORM_FILE, nalarmPicture,
		CURLFORM_FILENAME, nalarmPictureName,
		CURLFORM_CONTENTTYPE, "image/jpeg",
		CURLFORM_END);*/

	//设置会话内容
	curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:8000/monitor/alarm/");
	curl_easy_setopt(curl, CURLOPT_POST, 1);
	//curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
	//curl_easy_setopt(curl, CURLOPT_HEADERDATA, fp);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10);
	//curl_easy_setopt(curl, CURLOPT_COOKIEJAR, "C:\\Users\\18521\\Documents\\All_Project\\python\\cookies.txt");
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);
	//curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_ANY);
	//curl_easy_setopt(curl, CURLOPT_USERPWD, "Tom:9TOMweb");
	//curl_easy_setopt(curl, CURLOPT_LOGIN_OPTIONS, "AUTH=*");
	//curl_easy_setopt(curl, CURLOPT_USERNAME, "Tom");
	//curl_easy_setopt(curl, CURLOPT_PASSWORD, "9TOMweb");
	/*curl_easy_setopt(curl, CURLOPT_COPYPOSTFIELDS, szJsonData);*/

	//发送请求
	res = curl_easy_perform(curl);

	//结尾清理
	curl_formfree(formpost);
	//fclose(fp);
	curl_slist_free_all(headers);
	curl_easy_cleanup(curl);
	curl_global_cleanup();
	return res;
		
}

int HtpClient::Dotest() 
{
	CURLcode rescode = curl_global_init(CURL_GLOBAL_ALL);//建立程序环境
	CURL* curl = curl_easy_init();//得到curl_easy句柄
	struct curl_slist *headers = NULL;//请求头的容器
	CURLcode res;//状态码
	headers = curl_slist_append(headers, "Connect-Test: 1");
	curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:8000/monitor/alarm/test/");
	curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
	res = curl_easy_perform(curl);
	curl_slist_free_all(headers);
	curl_easy_cleanup(curl);
	curl_global_cleanup();
	return res;
}

int HtpClient::Doclient()
{
	CURLcode rescode = curl_global_init(CURL_GLOBAL_ALL);//建立程序环境
	CURL* curl = curl_easy_init();//得到curl_easy句柄
	struct curl_slist *headers = NULL;//请求头的容器
	CURLcode res;//状态码
	struct curl_httppost *formpost = NULL;
	struct curl_httppost *lastptr = NULL;
	curl_formadd(&formpost, &lastptr,
		CURLFORM_COPYNAME, "photo_alarm",
		CURLFORM_FILE, alarmPicture,
		CURLFORM_FILENAME, alarmPictureName,
		CURLFORM_CONTENTTYPE, "image/jpeg",
		CURLFORM_END);
	headers = curl_slist_append(headers, level);
	headers = curl_slist_append(headers, description);
	headers = curl_slist_append(headers, device);
	headers = curl_slist_append(headers, datetime);
	//headers = curl_slist_append(headers, alarmPicture);
	//headers = curl_slist_append(headers, alarmPictureName);
	curl_easy_setopt(curl, CURLOPT_URL, "http://192.168.43.25:8080/");
	//curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
	curl_easy_setopt(curl, CURLOPT_POST, 1);
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);
	res = curl_easy_perform(curl);
	curl_formfree(formpost);
	curl_slist_free_all(headers);
	curl_easy_cleanup(curl);
	curl_global_cleanup();
	return res;
}

HtpClient::~HtpClient()
{
}
