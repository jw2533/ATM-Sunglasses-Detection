#pragma once
#include <windows.h>
#include <wininet.h>
#pragma comment(lib,"wininet.lib")
 
#include <stdio.h>  
#include "curl/curl.h"    
//#pragma comment(lib,"libcurl_a_debug.lib") 
#include <string>
using namespace std;

class HtpClient
{
public:
	HtpClient();
	~HtpClient();
	//bool Doget();
	const char* level;
	const char* description;
	const char* device;
	const char* datetime;
	const char* alarmPicture;
	const char* alarmPictureName;
	int Dopost();
	int Dotest();
	int Doclient();
};

