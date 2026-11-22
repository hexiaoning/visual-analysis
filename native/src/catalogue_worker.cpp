#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

#include <Windows.h>
#include <wincrypt.h>
#include <winhttp.h>
#include <nlohmann/json.hpp>

namespace {

using Json = nlohmann::json;
constexpr std::array sources{"miit","dongchedi"};
constexpr std::array fields{"brand","series","model","year","trim","catalogue_id"};
constexpr size_t responseLimit=8*1024*1024;
constexpr wchar_t userAgent[]=L"VisualAnalysis/2.13 (https://github.com/hexiaoning/visual-analysis)";

struct Http {
    HINTERNET value{};
    explicit Http(HINTERNET handle):value(handle) { if(!handle) throw std::runtime_error("WinHTTP failed."); }
    ~Http(){WinHttpCloseHandle(value);}
    Http(const Http&)=delete;
    Http& operator=(const Http&)=delete;
};

std::string cleaned(const Json& record,const char* name,size_t limit=200) {
    if(!record.contains(name) || !record[name].is_string()) return {};
    auto value=record[name].get<std::string>();
    const auto first=value.find_first_not_of(" \t\r\n");
    if(first==std::string::npos) return {};
    value=value.substr(first,value.find_last_not_of(" \t\r\n")-first+1);
    if(value.size()>limit) {
        auto length=limit;
        while(length>0 && (static_cast<unsigned char>(value[length])&0xc0)==0x80) --length;
        value.resize(length);
    }
    return value;
}

std::wstring wide(const std::string& text) {
    const auto length=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,text.data(),int(text.size()),nullptr,0);
    if(length<=0) return {};
    std::wstring result(length,L'\0');
    MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,text.data(),int(text.size()),result.data(),length);
    return result;
}

struct Endpoint {std::wstring host,path; INTERNET_PORT port{};};
std::optional<Endpoint> parseEndpoint(const std::string& value) {
    if(value.empty() || value.find('\0')!=std::string::npos) return std::nullopt;
    const auto input=wide(value);
    URL_COMPONENTS parts{};
    parts.dwStructSize=sizeof(parts);
    parts.dwSchemeLength=parts.dwHostNameLength=parts.dwUserNameLength=parts.dwPasswordLength=
        parts.dwUrlPathLength=parts.dwExtraInfoLength=DWORD(-1);
    if(input.empty() || !WinHttpCrackUrl(input.c_str(),DWORD(input.size()),0,&parts) ||
       parts.nScheme!=INTERNET_SCHEME_HTTPS || parts.dwHostNameLength==0 || parts.dwUserNameLength ||
       parts.dwPasswordLength || input.find(L'#')!=std::wstring::npos) return std::nullopt;
    Endpoint output{std::wstring(parts.lpszHostName,parts.dwHostNameLength),
        parts.dwUrlPathLength ? std::wstring(parts.lpszUrlPath,parts.dwUrlPathLength):L"/",parts.nPort};
    if(parts.dwExtraInfoLength) output.path.append(parts.lpszExtraInfo,parts.dwExtraInfoLength);
    return output;
}

std::wstring tokenFor(std::string_view source) {
    const auto* name=source=="miit"?L"VISUAL_VEHICLE_MIIT_TOKEN":L"VISUAL_VEHICLE_DONGCHEDI_TOKEN";
    const auto size=GetEnvironmentVariableW(name,nullptr,0);
    if(!size || size>16000) return {};
    std::wstring value(size,L'\0');
    const auto length=GetEnvironmentVariableW(name,value.data(),size);
    if(!length || length>=size) return {};
    value.resize(length);
    if(value.find_first_of(L"\r\n")!=std::wstring::npos) return {};
    return value;
}

std::string timestamp() {
    SYSTEMTIME t{};
    GetSystemTime(&t);
    std::array<char,32> out{};
    std::snprintf(out.data(),out.size(),"%04u-%02u-%02uT%02u:%02u:%02u+00:00",
        t.wYear,t.wMonth,t.wDay,t.wHour,t.wMinute,t.wSecond);
    return out.data();
}

std::string encodeComponent(std::string_view text) {
    constexpr char hex[]="0123456789ABCDEF";
    std::string out;
    for(const unsigned char c:text) {
        if((c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') || c=='-' || c=='_' || c=='.') {
            out+=c;
        } else {
            out+='%';out+=hex[c>>4];out+=hex[c&15];
        }
    }
    return out;
}

std::string publicGet(const Endpoint& endpoint,size_t limit) {
    Http session(WinHttpOpen(userAgent,WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,nullptr,nullptr,0));
    if(!WinHttpSetTimeouts(session.value,5000,5000,6000,6000)) throw std::runtime_error("WinHTTP timeout setup failed.");
    Http connection(WinHttpConnect(session.value,endpoint.host.c_str(),endpoint.port,0));
    Http request(WinHttpOpenRequest(connection.value,L"GET",endpoint.path.c_str(),nullptr,nullptr,nullptr,
                                   WINHTTP_FLAG_SECURE|WINHTTP_FLAG_ESCAPE_DISABLE));
    DWORD redirects=WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
    if(!WinHttpSetOption(request.value,WINHTTP_OPTION_REDIRECT_POLICY,&redirects,sizeof(redirects)) ||
       !WinHttpSendRequest(request.value,L"Cache-Control: no-cache\r\n",DWORD(-1),nullptr,0,0,0) ||
       !WinHttpReceiveResponse(request.value,nullptr)) throw std::runtime_error("Public source did not respond.");
    DWORD status{},size=sizeof(status);
    if(!WinHttpQueryHeaders(request.value,WINHTTP_QUERY_STATUS_CODE|WINHTTP_QUERY_FLAG_NUMBER,nullptr,&status,&size,nullptr)
        || status<200 || status>=300) throw std::runtime_error("Public source returned an HTTP error.");
    std::string body;
    std::array<char,32768> buffer{};
    for (;;) {
        DWORD received{};
        if(!WinHttpReadData(request.value,buffer.data(),DWORD(buffer.size()),&received)) throw std::runtime_error("Public response interrupted.");
        if(!received) return body;
        if(received>limit-body.size()) throw std::runtime_error("Public response is too large.");
        body.append(buffer.data(),received);
    }
}

std::string normalized(std::string_view text) {
    std::string result;
    for(unsigned char c:text) {
        if(c>='A' && c<='Z') result+=static_cast<char>(c-'A'+'a');
        else if(c>=128 || (c>='a' && c<='z') || (c>='0' && c<='9')) result+=c;
    }
    return result;
}

std::string marketMake(std::string_view brand) {
    constexpr std::array aliases{
        std::pair{"比亚迪","BYD"},std::pair{"小米","Xiaomi"},std::pair{"理想","Li Auto"},std::pair{"问界","AITO"},
        std::pair{"吉利","Geely"},std::pair{"五菱","Wuling"},std::pair{"特斯拉","Tesla"},std::pair{"丰田","Toyota"},
        std::pair{"本田","Honda"},std::pair{"大众","Volkswagen"},std::pair{"福特","Ford"},std::pair{"日产","Nissan"},
        std::pair{"奔驰","Mercedes-Benz"},std::pair{"宝马","BMW"},std::pair{"奥迪","Audi"},std::pair{"现代","Hyundai"},
    };
    for(const auto& [key,value]:aliases) if(brand==key) return value;
    return std::string(brand);
}

std::string marketSeries(std::string_view series) {
    constexpr std::array aliases{
        std::pair{"海鸥","Seagull"},std::pair{"秦 PLUS","Qin Plus"},std::pair{"宋 PLUS","Song Plus"},
        std::pair{"汉","Han"},std::pair{"星愿","Xingyuan"},std::pair{"宏光 MINIEV","Hongguang Mini EV"},
        std::pair{"卡罗拉","Corolla"},std::pair{"凯美瑞","Camry"},std::pair{"思域","Civic"},std::pair{"朗逸","Lavida"},
    };
    for(const auto& [key,value]:aliases) if(series==key) return value;
    return std::string(series);
}

Json publicRecord(const Json& candidate,const std::string& title,const std::string& url,const std::string& id) {
    return {{"brand",candidate["brand"]},{"series",candidate["series"]},{"model",title},{"year",""},{"trim",""},
            {"catalogue_id",""},{"record_id",id},{"source_url",url},{"image_jpeg_base64",Json::array()}};
}

std::string referenceJpeg(const std::string& url) {
    try {
        const auto endpoint=parseEndpoint(url);
        if(!endpoint || (endpoint->host!=L"upload.wikimedia.org" && endpoint->host!=L"thumb.wikimedia.org") ||
           !endpoint->path.starts_with(L"/wikipedia/")) return {};
        const auto data=publicGet(*endpoint,512*1024);
        if(data.size()<3 || static_cast<unsigned char>(data[0])!=0xff || static_cast<unsigned char>(data[1])!=0xd8) return {};
        DWORD length{};
        const auto format=CRYPT_STRING_BASE64|CRYPT_STRING_NOCRLF;
        const auto* bytes=reinterpret_cast<const BYTE*>(data.data());
        if(!CryptBinaryToStringA(bytes,DWORD(data.size()),format,nullptr,&length)) return {};
        std::string encoded(length,'\0');
        if(!CryptBinaryToStringA(bytes,DWORD(data.size()),format,encoded.data(),&length)) return {};
        encoded.resize(length);
        return encoded;
    } catch(const std::exception&) { return {}; }
}

Json wikiLookup(const Json& candidates) {
    Json result={{"source","wikimedia"},{"queried_at",timestamp()},{"status","unavailable"},{"records",Json::array()}};
    if(candidates.empty()) {result["status"]="no_candidate";return result;}
    bool responded=false;
    std::vector<bool> matched(candidates.size(),false);
    std::vector<std::pair<size_t,std::string>> thumbs;
    for(const auto* domain:{"en.wikipedia.org","zh.wikipedia.org"}) {
        std::vector<std::pair<size_t,std::string>> titles;
        std::string list;
        for(size_t i=0;i<candidates.size();++i) {
            if(matched[i]) continue;
            const auto brand=cleaned(candidates[i],"brand"),series=cleaned(candidates[i],"series");
            auto title=domain[0]=='e' ? marketMake(brand)+" "+marketSeries(series) : brand+series;
            if(title.empty()) continue;
            if(!list.empty()) list+="%7C";
            list+=encodeComponent(title);
            titles.emplace_back(i,std::move(title));
        }
        if(titles.empty()) continue;
        try {
            const auto tail="/w/api.php?action=query&formatversion=2&format=json&redirects=1"
                "&prop=pageimages%7Cinfo&inprop=url&piprop=thumbnail%7Cname&pithumbsize=480&pilicense=free&titles="+list;
            const auto value=Json::parse(publicGet({wide(domain),wide(tail),443},512*1024));
            const auto& query=value.at("query");
            const auto& pages=query.at("pages");
            responded=true;
            for(const auto& [index,title]:titles) {
                const auto expected=normalized(title);
                auto redirected=expected;
                for(const auto& mapping:query.value("redirects",Json::array())) {
                    if(normalized(cleaned(mapping,"from"))==redirected) redirected=normalized(cleaned(mapping,"to"));
                }
                for(const auto& page:pages) {
                    if(!page.contains("pageid")) continue;
                    const auto pageTitle=cleaned(page,"title"), canonical=normalized(pageTitle);
                    // Redirects to a generic brand or unrelated short title do not establish a model.
                    if(canonical.empty() || (canonical!=expected &&
                       !(canonical==redirected && canonical.starts_with(expected)))) continue;
                    const auto link=cleaned(page,"fullurl",2048);
                    const auto endpoint=parseEndpoint(link);
                    if(!endpoint || endpoint->host!=wide(domain)) continue;
                    auto record=publicRecord(candidates[index],pageTitle,link,std::to_string(page.at("pageid").get<int64_t>()));
                    if(page.contains("thumbnail")) {
                        const auto image=cleaned(page["thumbnail"],"source",2048);
                        const auto filename=cleaned(page,"pageimage",1024);
                        if(!image.empty()) {
                            thumbs.emplace_back(result["records"].size(),image);
                            record["reference_image_url"]=image;
                            if(!filename.empty()) record["image_license_page"]="https://commons.wikimedia.org/wiki/File:"+encodeComponent(filename);
                        }
                    }
                    result["records"].push_back(std::move(record));
                    matched[index]=true;
                    break;
                }
            }
        } catch(const std::exception&) {}
    }
    const auto fetch=[&](size_t start) {
        std::vector<std::pair<size_t,std::string>> downloaded;
        for(size_t index=start;index<thumbs.size();index+=3) {
            auto image=referenceJpeg(thumbs[index].second);
            if(!image.empty()) downloaded.emplace_back(thumbs[index].first,std::move(image));
        }
        return downloaded;
    };
    auto first=std::async(std::launch::async,fetch,0);
    auto second=std::async(std::launch::async,fetch,1);
    const auto collect=[&](const auto& images){
        for(const auto& [index,picture]:images) result["records"][index]["image_jpeg_base64"].push_back(picture);
    };
    collect(fetch(2));collect(first.get());collect(second.get());
    for(auto& record:result["records"]) if(record["image_jpeg_base64"].empty()) {
        record.erase("reference_image_url");
        record.erase("image_license_page");
    }
    if(responded) result["status"]=result["records"].empty()?"no_match":"ok";
    return result;
}

Json vpicLookup(const Json& candidates) {
    Json result={{"source","nhtsa_vpic"},{"queried_at",timestamp()},{"status","unavailable"},{"records",Json::array()}};
    if(candidates.empty()) {result["status"]="no_candidate";return result;}
    bool responded=false;
    std::map<std::string,Json> received;
    for(const auto& candidate:candidates) {
        const auto make=marketMake(cleaned(candidate,"brand")), series=marketSeries(cleaned(candidate,"series"));
        if(make.empty() || series.empty()) continue;
        const auto tail="/api/vehicles/GetModelsForMake/"+encodeComponent(make)+"?format=json";
        try {
            auto found=received.find(make);
            if(found==received.end()) {
                auto fetched=Json::parse(publicGet({L"vpic.nhtsa.dot.gov",wide(tail),443},1024*1024)).at("Results");
                found=received.emplace(make,std::move(fetched)).first;
            }
            if(!found->second.is_array()) continue;
            responded=true;
            for(const auto& match:found->second) {
                if(normalized(cleaned(match,"Make_Name"))!=normalized(make)
                    || normalized(cleaned(match,"Model_Name"))!=normalized(series)) continue;
                auto record=publicRecord(candidate,cleaned(match,"Model_Name"),"https://vpic.nhtsa.dot.gov"+tail,
                    std::to_string(match.at("Model_ID").get<int64_t>()));
                result["records"].push_back(std::move(record));
                break;
            }
        } catch(const std::exception&) {}
    }
    if(responded) result["status"]=result["records"].empty()?"no_match":"ok";
    return result;
}

Json parseRecords(const Json& payload) {
    const auto& records=payload.at("records");
    if(!records.is_array()) throw std::invalid_argument("Invalid catalogue response.");
    auto result=Json::array();
    for(const auto& item:records) {
        if(result.size()>=12) break;
        if(!item.is_object()) throw std::invalid_argument("Invalid catalogue item.");
        Json record=Json::object();
        for(const auto* key:fields) record[key]=cleaned(item,key);
        record["record_id"]=cleaned(item,"record_id");
        const auto url=cleaned(item,"source_url",2048);
        record["source_url"]=parseEndpoint(url)?url:"";
        if(record["series"]=="" || (record["source_url"]=="" && record["record_id"]=="")) {
            throw std::invalid_argument("Catalogue provenance is missing.");
        }
        record["parameters"]=Json::object();
        if(item.contains("parameters") && item["parameters"].is_object()) {
            for(auto part=item["parameters"].begin();part!=item["parameters"].end() && record["parameters"].size()<12;++part) {
                if(part.key().size()<=80 && part.value().is_string()) record["parameters"][part.key()]=cleaned(item["parameters"],part.key().c_str());
            }
        }
        record["image_jpeg_base64"]=Json::array();
        if(item.contains("image_jpeg_base64") && item["image_jpeg_base64"].is_array()) {
            for(const auto& picture:item["image_jpeg_base64"]) {
                if(record["image_jpeg_base64"].size()==2) break;
                if(picture.is_string() && picture.get_ref<const std::string&>().size()<=1024*1024) record["image_jpeg_base64"].push_back(picture);
            }
        }
        result.push_back(std::move(record));
    }
    return result;
}

Json lookup(const char* source,const Json& settings,const Json& candidates) {
    Json result={{"source",source},{"queried_at",""},{"status","not_configured"},{"records",Json::array()}};
    if(candidates.empty()) {result["status"]="no_candidate";return result;}
    if(!settings.is_object() || !settings.contains(source) || !settings[source].is_object()) return result;
    const auto endpoint=parseEndpoint(cleaned(settings[source],"url",2048));
    const auto token=tokenFor(source);
    if(!endpoint || token.empty()) return result;
    result["queried_at"]=timestamp();
    result["status"]="unavailable";
    try {
        Http session(WinHttpOpen(userAgent,WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,nullptr,nullptr,0));
        if(!WinHttpSetTimeouts(session.value,8000,8000,8000,8000)) return result;
        Http connection(WinHttpConnect(session.value,endpoint->host.c_str(),endpoint->port,0));
        Http request(WinHttpOpenRequest(connection.value,L"POST",endpoint->path.c_str(),nullptr,nullptr,nullptr,WINHTTP_FLAG_SECURE));
        DWORD redirects=WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
        if(!WinHttpSetOption(request.value,WINHTTP_OPTION_REDIRECT_POLICY,&redirects,sizeof(redirects))) return result;
        std::wstring headers=L"Content-Type: application/json\r\nAccept: application/json\r\nCache-Control: no-store, no-cache\r\nAuthorization: Bearer "+token+L"\r\n";
        auto body=Json({{"schema_version",1},{"source",source},{"candidates",candidates}}).dump();
        if(!WinHttpSendRequest(request.value,headers.c_str(),DWORD(headers.size()),body.data(),
            DWORD(body.size()),DWORD(body.size()),0) || !WinHttpReceiveResponse(request.value,nullptr)) return result;
        DWORD status{},size=sizeof(status);
        if(!WinHttpQueryHeaders(request.value,WINHTTP_QUERY_STATUS_CODE|WINHTTP_QUERY_FLAG_NUMBER,nullptr,&status,&size,nullptr)) return result;
        if(status==401 || status==403) {result["status"]="unauthorized";return result;}
        if(status<200 || status>=300) return result;
        std::string content;
        std::array<char,65536> buffer{};
        for (;;) {
            DWORD received{};
            if(!WinHttpReadData(request.value,buffer.data(),DWORD(buffer.size()),&received)) return result;
            if(!received) break;
            content.append(buffer.data(),received);
            if(content.size()>responseLimit) {result["status"]="invalid_response";return result;}
        }
        result["records"]=parseRecords(Json::parse(content));
        result["status"]=result["records"].empty()?"no_match":"ok";
    } catch(const Json::exception&) {result["status"]="invalid_response";}
      catch(const std::invalid_argument&) {result["status"]="invalid_response";}
      catch(const std::exception&) {}
    return result;
}

}

int wmain(int count,wchar_t** arguments) {
    if(count!=3) return 2;
    try {
        const std::filesystem::path requestFile(arguments[1]), responseFile(arguments[2]);
        if(std::filesystem::file_size(requestFile)>65536) return 2;
        std::ifstream input(requestFile,std::ios::binary);
        const auto request=Json::parse(input);
        const auto& settings=request.at("settings");
        auto candidates=Json::array();
        const auto sanitize=[](const Json& supplied,size_t limit){
            auto items=Json::array();
            for(const auto& candidate:supplied) {
                if(items.size()==limit) break;
                if(!candidate.is_object() || cleaned(candidate,"series").empty()) continue;
                auto safety=Json::object();
                for(const auto* field:fields) safety[field]=cleaned(candidate,field);
                items.push_back(std::move(safety));
            }
            return items;
        };
        candidates=sanitize(request.at("candidates"),3);
        auto publicCandidates=sanitize(request.value("public_candidates",candidates),32);
        auto wiki=std::async(std::launch::async,wikiLookup,std::cref(publicCandidates));
        auto vpic=std::async(std::launch::async,vpicLookup,std::cref(candidates));
        auto first=std::async(std::launch::async,lookup,sources[0],std::cref(settings),std::cref(candidates));
        auto second=lookup(sources[1],settings,candidates);
        auto results=Json::array({wiki.get(),vpic.get()});
        auto authorized=first.get();
        if(authorized["status"]!="not_configured" && authorized["status"]!="no_candidate") results.push_back(std::move(authorized));
        if(second["status"]!="not_configured" && second["status"]!="no_candidate") results.push_back(std::move(second));
        std::ofstream output(responseFile,std::ios::binary|std::ios::trunc);
        output<<results.dump()<<'\n';
        output.close();
        return output?0:2;
    } catch(const std::exception&) {return 2;}
}
