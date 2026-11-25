#include "incident_screen.h"
#include "local_images.h"

#include <Windows.h>
#include <msopc.h>
#include <msxml6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <map>
#include <optional>
#include <set>
#include <sstream>

namespace visual {
namespace {
namespace fs=std::filesystem;
using Microsoft::WRL::ComPtr;

constexpr const wchar_t* relationshipPrefix=L"http://schemas.openxmlformats.org/officeDocument/2006/relationships/";
constexpr std::string_view sheetNamespace="http://schemas.openxmlformats.org/spreadsheetml/2006/main";

void check(HRESULT status,const char* message) {
    if(FAILED(status)) throw std::runtime_error(message);
}

struct Apartment {
    Apartment() {
        const auto result=CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED);
        check(result==RPC_E_CHANGED_MODE?S_OK:result,"无法初始化 Excel 工作簿组件。");
        uninitialize=SUCCEEDED(result);
    }
    ~Apartment() {if(uninitialize) CoUninitialize();}
    bool uninitialize{};
};

struct Bstr {
    BSTR value{};
    Bstr()=default;
    explicit Bstr(const wchar_t* text):value(SysAllocString(text)) {
        if(!value) throw std::bad_alloc();
    }
    ~Bstr() {SysFreeString(value);}
    std::wstring text() const {return value?std::wstring(value,SysStringLen(value)):std::wstring{};}
};

ComPtr<IOpcPart> relatedPart(IOpcPackage* package,IOpcRelationship* relation) {
    OPC_URI_TARGET_MODE mode{};
    check(relation->GetTargetMode(&mode),"无法读取工作簿关系。");
    if(mode!=OPC_URI_TARGET_MODE_INTERNAL) throw std::invalid_argument("工作簿必须引用本地工作表。");
    ComPtr<IOpcUri> source;
    ComPtr<IUri> target;
    ComPtr<IOpcPartUri> part;
    check(relation->GetSourceUri(&source),"无法读取工作簿部件地址。");
    check(relation->GetTargetUri(&target),"无法读取工作簿目标地址。");
    check(source->CombinePartUri(target.Get(),&part),"工作簿部件地址无效。");
    ComPtr<IOpcPartSet> parts;
    check(package->GetPartSet(&parts),"无法读取工作簿部件。");
    ComPtr<IOpcPart> result;
    check(parts->GetPart(part.Get(),&result),"工作簿部件缺失。");
    return result;
}

ComPtr<IOpcPart> partByType(IOpcPackage* package,IOpcRelationshipSet* relations,const wchar_t* kind) {
    const auto type=std::wstring(relationshipPrefix)+kind;
    ComPtr<IOpcRelationshipEnumerator> items;
    check(relations->GetEnumeratorForType(type.c_str(),&items),"无法读取工作簿关系。");
    BOOL hasNext=FALSE;
    check(items->MoveNext(&hasNext),"工作簿关系无效。");
    if(!hasNext) return {};
    ComPtr<IOpcRelationship> relation;
    check(items->GetCurrent(&relation),"无法读取工作簿关系。");
    return relatedPart(package,relation.Get());
}

ComPtr<IStream> contents(IOpcPart* part) {
    ComPtr<IStream> stream;
    check(part->GetContentStream(&stream),"无法读取工作簿内容。");
    STATSTG info{};
    check(stream->Stat(&info,STATFLAG_NONAME),"无法读取工作簿部件大小。");
    if(info.cbSize.QuadPart>128ULL*1024*1024) throw std::invalid_argument("Excel 部件超过可读取大小。");
    return stream;
}

ComPtr<IXMLDOMDocument3> xml(IOpcPart* part) {
    auto stream=contents(part);
    ComPtr<IXMLDOMDocument3> document;
    check(CoCreateInstance(__uuidof(DOMDocument60),nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&document)),
          "无法初始化 Excel XML 读取组件。");
    check(document->put_async(VARIANT_FALSE),"Excel XML 初始化失败。");
    check(document->put_validateOnParse(VARIANT_FALSE),"Excel XML 初始化失败。");
    check(document->put_resolveExternals(VARIANT_FALSE),"Excel XML 初始化失败。");
    check(document->put_preserveWhiteSpace(VARIANT_TRUE),"Excel XML 初始化失败。");
    VARIANT setting{};
    setting.vt=VT_BOOL;
    setting.boolVal=VARIANT_TRUE;
    Bstr prohibit(L"ProhibitDTD");
    check(document->setProperty(prohibit.value,setting),"Excel XML 初始化失败。");
    VARIANT input{};
    input.vt=VT_UNKNOWN;
    input.punkVal=stream.Get();
    VARIANT_BOOL loaded{};
    check(document->load(input,&loaded),"Excel 工作表 XML 不可读。");
    if(loaded!=VARIANT_TRUE) throw std::invalid_argument("Excel 工作表 XML 无效。");
    return document;
}

ComPtr<IXMLDOMNodeList> nodes(IXMLDOMNode* node,const wchar_t* path) {
    ComPtr<IXMLDOMNodeList> result;
    Bstr query(path);
    check(node->selectNodes(query.value,&result),"Excel 工作表结构无效。");
    return result;
}

ComPtr<IXMLDOMNode> node(IXMLDOMNode* owner,const wchar_t* path) {
    ComPtr<IXMLDOMNode> result;
    Bstr query(path);
    check(owner->selectSingleNode(query.value,&result),"Excel 工作表结构无效。");
    return result;
}

std::wstring nodeText(IXMLDOMNode* element) {
    if(!element) return {};
    Bstr value;
    check(element->get_text(&value.value),"Excel 单元格不可读。");
    return value.text();
}

std::wstring attribute(IXMLDOMNode* element,const wchar_t* name) {
    ComPtr<IXMLDOMNamedNodeMap> attributes;
    check(element->get_attributes(&attributes),"Excel 节点属性不可读。");
    if(!attributes) return {};
    Bstr key(name);
    ComPtr<IXMLDOMNode> property;
    check(attributes->getNamedItem(key.value,&property),"Excel 节点属性无效。");
    return nodeText(property.Get());
}

std::wstring textRuns(IXMLDOMNode* element) {
    const auto list=nodes(element,L"./*[local-name()='t'] | ./*[local-name()='r']/*[local-name()='t']");
    std::wstring result;
    for(;;) {
        ComPtr<IXMLDOMNode> entry;
        check(list->nextNode(&entry),"Excel 字符串不可读。");
        if(!entry) break;
        result+=nodeText(entry.Get());
    }
    return result;
}

std::vector<std::string> sharedStrings(IOpcPart* part) {
    if(!part) return {};
    const auto document=xml(part);
    const auto list=nodes(document.Get(),L"/*[local-name()='sst']/*[local-name()='si']");
    std::vector<std::string> result;
    for(;;) {
        ComPtr<IXMLDOMNode> entry;
        check(list->nextNode(&entry),"Excel 共享字符串不可读。");
        if(!entry) break;
        result.push_back(utf8(textRuns(entry.Get())));
    }
    return result;
}

size_t columnIndex(std::wstring_view cell) {
    size_t index=0;
    for(const auto ch:cell) {
        if(ch<L'A' || ch>L'Z') break;
        index=index*26+size_t(ch-L'A'+1);
        if(index>16384) throw std::invalid_argument("Excel 列超出范围。");
    }
    if(!index) throw std::invalid_argument("Excel 单元格列号无效。");
    return index-1;
}

std::string columnName(size_t index) {
    std::string name;
    for(++index;index;index=(index-1)/26) name.push_back(static_cast<char>('A'+(index-1)%26));
    std::reverse(name.begin(),name.end());
    return name;
}

size_t countValue(const std::string& input) {
    size_t value{};
    const auto parsed=std::from_chars(input.data(),input.data()+input.size(),value);
    if(parsed.ec!=std::errc{} || parsed.ptr!=input.data()+input.size()) throw std::invalid_argument("Excel 单元格索引无效。");
    return value;
}

struct Cell {
    size_t column{};
    std::string value,type,style;
};
struct Row {
    size_t original{};
    std::vector<Cell> values;
};
struct Worksheet {
    std::vector<Row> rows;
    std::map<std::string,size_t> headers;
    size_t columns{};
    bool date1904{};
    std::string styleXml;
};

std::string readStyle(IOpcPart* part) {
    if(!part) return {};
    const auto stream=contents(part);
    STATSTG info{};
    check(stream->Stat(&info,STATFLAG_NONAME),"无法读取 Excel 样式。");
    std::string result(static_cast<size_t>(info.cbSize.QuadPart),'\0');
    ULONG read{};
    check(stream->Read(result.data(),static_cast<ULONG>(result.size()),&read),"无法读取 Excel 样式。");
    if(read!=result.size()) throw std::runtime_error("Excel 样式读取不完整。");
    return result;
}

std::string_view valueAt(const Row& row,size_t column) {
    const auto cell=std::lower_bound(row.values.begin(),row.values.end(),column,
        [](const Cell& item,size_t index){return item.column<index;});
    return cell!=row.values.end() && cell->column==column?cell->value:std::string_view{};
}

std::string trimmed(std::string_view text) {
    const auto first=text.find_first_not_of(" \t\r\n");
    if(first==std::string_view::npos) return {};
    return std::string(text.substr(first,text.find_last_not_of(" \t\r\n")-first+1));
}

bool hasHeaders(const std::map<std::string,size_t>& headers) {
    return headers.contains("报警时间") && headers.contains("报警类别") && headers.contains("报警类型") &&
        headers.contains("报警细类") && headers.contains("事件详情");
}

Worksheet readWorksheet(IOpcPart* part,const std::vector<std::string>& strings,const std::atomic<bool>& cancel) {
    const auto document=xml(part);
    const auto rows=nodes(document.Get(),L"/*[local-name()='worksheet']/*[local-name()='sheetData']/*[local-name()='row']");
    Worksheet result;
    size_t scanned=0, previous=0;
    for(;;) {
        if(cancel) return {};
        ComPtr<IXMLDOMNode> rowNode;
        check(rows->nextNode(&rowNode),"Excel 行不可读。");
        if(!rowNode) break;
        const auto r=attribute(rowNode.Get(),L"r");
        Row row{r.empty()?previous+1:countValue(utf8(r)),{}};
        if(row.original<=previous || row.original>1048576) throw std::invalid_argument("Excel 行位置无效。");
        previous=row.original;
        const auto cells=nodes(rowNode.Get(),L"./*[local-name()='c']");
        size_t nextColumn=0;
        for(;;) {
            ComPtr<IXMLDOMNode> cellNode;
            check(cells->nextNode(&cellNode),"Excel 单元格不可读。");
            if(!cellNode) break;
            const auto reference=attribute(cellNode.Get(),L"r");
            const auto column=reference.empty()?nextColumn:columnIndex(reference);
            nextColumn=column+1;
            auto type=utf8(attribute(cellNode.Get(),L"t"));
            auto value=utf8(nodeText(node(cellNode.Get(),L"./*[local-name()='v']").Get()));
            if(type=="s") {
                const auto index=countValue(value);
                if(index>=strings.size()) throw std::invalid_argument("Excel 共享字符串索引超出范围。");
                value=strings[index];
                type="inlineStr";
            } else if(type=="inlineStr") {
                const auto inlineValue=node(cellNode.Get(),L"./*[local-name()='is']");
                if(inlineValue) value=utf8(textRuns(inlineValue.Get()));
            } else if(type=="str") {
                type="inlineStr";
            }
            row.values.push_back({column,std::move(value),std::move(type),utf8(attribute(cellNode.Get(),L"s"))});
        }
        std::ranges::sort(row.values,{},&Cell::column);
        if(result.headers.empty()) {
            std::map<std::string,size_t> headers;
            for(const auto& cell:row.values) if(!cell.value.empty()) headers.try_emplace(trimmed(cell.value),cell.column);
            if(!hasHeaders(headers)) {
                if(++scanned>=24) break;
                continue;
            }
            result.headers=std::move(headers);
        }
        if(!row.values.empty()) result.columns=std::max(result.columns,row.values.back().column+1);
        if(result.rows.empty() || std::ranges::any_of(row.values,[](const Cell& cell){return !cell.value.empty();})) {
            result.rows.push_back(std::move(row));
        }
    }
    return result;
}

Worksheet readWorkbook(IOpcFactory* factory,const fs::path& source,const std::atomic<bool>& cancel) {
    if(_wcsicmp(source.extension().c_str(),L".xlsx")!=0 || !fs::is_regular_file(source)) {
        throw std::invalid_argument("请选择 110 接警系统导出的 .xlsx 文件。");
    }
    if(fs::file_size(source)>96ULL*1024*1024) throw std::invalid_argument("警情 Excel 超过 96 MiB。");
    ComPtr<IStream> file;
    check(factory->CreateStreamOnFile(source.c_str(),OPC_STREAM_IO_READ,nullptr,FILE_ATTRIBUTE_NORMAL,&file),
          "无法打开警情 Excel。");
    ComPtr<IOpcPackage> package;
    check(factory->ReadPackageFromStream(file.Get(),OPC_CACHE_ON_ACCESS,&package),"无法打开警情 Excel 工作簿。");
    ComPtr<IOpcRelationshipSet> rootRelations;
    check(package->GetRelationshipSet(&rootRelations),"无法读取工作簿关系。");
    const auto workbook=partByType(package.Get(),rootRelations.Get(),L"officeDocument");
    if(!workbook) throw std::invalid_argument("Excel 工作簿缺少主文档。");
    ComPtr<IOpcRelationshipSet> relations;
    check(workbook->GetRelationshipSet(&relations),"无法读取工作表关系。");
    const auto meta=xml(workbook.Get());
    const auto settings=node(meta.Get(),L"/*[local-name()='workbook']/*[local-name()='workbookPr']");
    const auto date1904=settings?attribute(settings.Get(),L"date1904"):std::wstring{};
    const auto strings=sharedStrings(partByType(package.Get(),relations.Get(),L"sharedStrings").Get());
    const auto worksheets=nodes(meta.Get(),L"/*[local-name()='workbook']/*[local-name()='sheets']/*[local-name()='sheet']");
    for(;;) {
        if(cancel) return {};
        ComPtr<IXMLDOMNode> sheet;
        check(worksheets->nextNode(&sheet),"无法读取 Excel 工作表列表。");
        if(!sheet) break;
        const auto key=node(sheet.Get(),L"@*[local-name()='id']");
        if(!key) continue;
        ComPtr<IOpcRelationship> relation;
        check(relations->GetRelationship(nodeText(key.Get()).c_str(),&relation),"Excel 工作表关系无效。");
        LPWSTR rawType{};
        check(relation->GetRelationshipType(&rawType),"Excel 工作表关系无效。");
        const std::wstring type(rawType);
        CoTaskMemFree(rawType);
        if(!type.ends_with(L"/worksheet")) continue;
        auto result=readWorksheet(relatedPart(package.Get(),relation.Get()).Get(),strings,cancel);
        if(result.headers.empty()) continue;
        result.date1904=date1904==L"1" || date1904==L"true";
        result.styleXml=readStyle(partByType(package.Get(),relations.Get(),L"styles").Get());
        return result;
    }
    if(cancel) return {};
    throw std::invalid_argument("找不到含「报警时间、报警类别、报警类型、报警细类、事件详情」的表头。");
}

Json defaultRules() {
    return Json::parse(R"json(
{
  "version": 1,
  "review_keywords": ["查无此事","查无实据","报警内容不实","报警内容不属实","虚假举报","未发现举报所述","未见举报所述","警情不属实","经核实不属实","经查不属实","未发现相关违法","并未被盗","没有被盗","不是被盗","并无盗窃","并非诈骗","未发生诈骗","不是诈骗","误报","谎报","虚报警情","虚假警情","重复报警","重复警情","重复拨打","来电咨询","电话咨询","仅咨询","咨询如何","落在车上","遗落在车上","已交回失主","已找到遗失物","误以为","该警情不属实","举报内容不属实","查明系误报","经核实系误会","报警事实不符"],
  "outcome_review_keywords": ["系误会","系误报","经查属误报","举报不实","核实为误报","查实为误报","查明未发生","未发现违法事实","无相关违法行为"],
  "rules": [
    {"category":"入室盗窃",
     "context_targets":["盗窃","被盗","失窃","偷走","盗走","财物被拿走","财物完好"],
     "structured":["入室盗窃","入户盗窃","盗窃室内财物","盗窃居民住宅","盗窃住宅","入室偷盗"],
     "keywords":["入室盗窃","入户盗窃","入室偷盗","入户偷窃","翻窗入室","撬门入室","破门入室","家中被盗","家里被盗","屋内被盗","家中失窃","房间被盗","入室行窃","入户行窃","住宅被盗","闯入家中盗窃","进屋偷东西"],
     "all_of":[
       [["家里","家中","屋内","住宅","住处","房屋","卧室"],["进了人","进人","闯入","翻窗","撬门","门锁被撬","门锁不对劲","撬开房门","门锁损坏","门锁异常","翻乱","翻动"],["被盗","不见","丢失","失窃","盗走","被偷","被拿走"]],
       [["家里","家中","屋内","住宅","住处","房屋"],["潜入","进入","溜入","溜进"],["盗窃","行窃","偷走","盗走"]],
       [["金戒指","财物","现金","手提包","贵重物品"],["门锁不对劲","门锁异常","撬开房门","撬窗","撬锁"],["不见","被盗","失窃","丢失"]],
       [["家里","家中","卧室","住宅"],["窗户","窗子"],["翻出","翻进"],["被盗","失窃","偷走","盗走"]]
     ],
     "exclude":["盗窃车内财物","只是遗失","没有丢东西"]},
    {"category":"盗窃车内财物",
     "context_targets":["盗窃","被盗","失窃","偷走","盗走","财物被拿走","财物完好"],
     "structured":["盗窃车内财物","车内财物被盗","盗窃机动车内财物","砸车盗窃","拉车门盗窃","盗窃车内物品"],
     "keywords":["车内财物被盗","盗窃车内财物","盗窃车内物品","车内物品被盗","砸车盗窃","拉车门盗窃","从车里偷走","砸车窗盗窃","车内被盗","车内物品失窃"],
     "all_of":[
       [["车内","车里","汽车里","汽车内","汽车里面","车窗","车门","车玻璃"],["被砸","砸破","被撬","撬锁","翻乱","拉车门"],["被盗","失窃","不见","丢失","被拿走","盗走","被偷"]],
       [["车内","车里","汽车里","汽车内","汽车里面"],["钱","现金","财物","物品","钱包","背包","手机","香烟"],["被盗","失窃","不见","丢失","被拿走","盗走","被偷"]],
       [["车内","车里","车窗","车门"],["盗走","行窃","偷东西"]]
     ],
     "exclude":["车辆本身被盗","仅车窗受损"]},
    {"category":"扒窃",
     "context_targets":["扒窃","被盗","被偷","偷走","盗走","财物完好"],
     "structured":["扒窃","扒盗","拎包扒窃","公共场所扒窃"],
     "keywords":["扒窃","扒手","扒走","掏包","掏走手机","掏走钱包","从口袋偷走","从衣兜偷走","贴身偷走"],
     "all_of":[
       [["地铁","公交","车站","排队","拥挤","候车","骑车"],["口袋","衣兜","裤兜","手提袋","挎包","背包","随身","外套","手包"],["被偷","被盗","偷走","取走","被拿走"]],
       [["口袋","裤兜","衣兜","贴身","身上"],["手机","钱包","钱","证件"],["被偷","被盗","偷走","取走","掏走"]],
       [["背包","手提袋","挎包"],["拉开","划开","拉链"],["被盗","不见","扒走","偷走"]]
     ],
     "exclude":["手机可能遗失","钱包可能遗失","自行掉落","自己遗失","自己掉了","不确定是掉了还是被偷"]},
    {"category":"电信诈骗",
     "context_targets":["诈骗","受骗","被骗","骗取","转账","资金损失","资金被转","盗刷"],
     "structured":["电信诈骗","电信网络诈骗","网络诈骗","电诈","电信网络新型违法犯罪","网络购物诈骗","刷单诈骗","网上刷单诈骗","冒充客服诈骗","网络交友诈骗"],
     "keywords":["电信诈骗","电信网络诈骗","网络诈骗","电诈","刷单诈骗","网络刷单","虚假投资理财","冒充公检法","冒充客服","杀猪盘","虚假网络贷款","远程屏幕共享被骗","网购退款诈骗","网络刷单返利","网络交友诈骗","冒充熟人诈骗"],
     "all_of":[
       [["刷单","任务返现"],["转账","付款","充值","扫码"],["返利","返现","解冻"]],
       [["客服","信贷","贷款人员","公检法","银行工作人员"],["自称","冒充","假称"],["要求转账","诱导转账","按要求转账","要求打款","要求汇款","扫码转钱","扫码付款","扫码支付","向对方转账","转账给对方","给对方转账"]],
       [["客服","贷款","信贷"],["影响征信","刷流水","流水不足","保证金","解冻费","资产证明"],["转账","扫码","汇款","打款"]],
       [["客服电话","客服"],["退款","退费"],["屏幕共享","验证码","扫码","转账"],["骗取","被骗","账户被扣款","被盗刷","资金被转走","资金被转出","钱被转走","钱被转出","要求私下转账","诱导共享","要求共享","退款后拉黑"]],
       [["来电","接到电话","打电话","电话"],["客服","售后人员"],["自称","冒充","假称"],["按对方要求","按对方指示","按照指示","要求分次","要求转账","诱导转账"],["转账","汇款","扫码转钱"]],
       [["公安","民警","检察院","法院"],["冒充","自称","假称"],["涉嫌洗钱","资金清查","资金核查","资金审查","账户审查"],["共享屏幕","屏幕共享","远程控制"]],
       [["客服","售后人员"],["自称","冒充","假称"],["诱导操作","按指示操作","按照指示","按要求操作","远程操作","屏幕共享","共享屏幕","输入验证码","要求下载"],["资金被转走","资金被转出","钱被转走","钱被转出","账户被扣款","被盗刷"]],
       [["陌生电话","网络","微信","网购","网上"],["转账","付款","投资","输入验证码"],["被骗","无法提现","上当","诈骗"]],
       [["微信聊天","通过微信","通过手机微信","QQ联系","通过QQ","聊天软件","网上聊天","交友平台","征婚网站","通过电话","以电话","电话联系","短信联系","网络平台"],["冒充","假扮","假冒","捏造","编造","虚构","虚假身份","假称"],["转账","汇款","转钱","付款","骗取财物","骗取被害人","骗走"],["骗取","被骗","无法提现","断绝联系","切断联系","不再联系","拉黑"]],
       [["网上","微信","QQ","网购","电商","二手交易平台"],["低价出售","低价销售","发布虚假","假装出售","诱导私下交易"],["收款后不发货","付款后失联","骗取","被骗","拉黑"]],
       [["陌生短信","陌生来电","钓鱼链接","陌生链接"],["银行卡","验证码","屏幕共享","扫码支付"],["未经本人操作","资金被转走","账户被扣款","被骗","被盗刷"]],
       [["客服","售后人员"],["冒充","自称","假称"],["按对方指示操作","按来电者指示操作","依照对方指示","按对方要求操作"],["账户","银行卡","存款","资金"],["被转出","被转走"]],
       [["客服"],["退款","退费"],["屏幕共享","共享屏幕"],["验证码"],["银行卡","账户"],["被转走","被转出"]],
       [["微信","QQ","短信","电话"],["冒充","假冒","伪装"],["转账","打款","汇款"],["从未借钱","本人并未","本人否认","并非本人","冒充熟人"]]
     ],
     "review_all_of":[
       [["网上认识","网恋","在微信认识","陌陌","聊天平台","短视频交友","网络认识","交友软件","网络交友","社交软件","交友平台"],["自称能","谎称能","虚构事实","编造理由","虚假身份","冒充"],["骗取","要求汇款","要求转账","收款后失联","微信转账"]],
       [["客服电话","微信客服","陌生号码"],["屏幕共享","验证码","扫码支付"],["要求付款","要求转账","要求汇款","疑似被诈骗"]],
       [["代办","办证","办理证件"],["实名认证","身份核验","人脸识别"],["屏幕共享","共享屏幕"],["远程控制","远程操控","手机失控","手机黑屏"]]
     ],
     "exclude":["反诈宣传","收到反诈预警","防骗宣传","咨询反诈","反诈提醒","反诈演练","警方提醒谨防","警惕冒充","谨防冒充","防范电信诈骗","防范网络诈骗","只是咨询诈骗","诈骗案例","电话调解后双方","查明未遭诈骗","发现未遭诈骗"],
     "outcome_exclude":["电话调解","合同纠纷","借贷纠纷","退款纠纷","退费纠纷","婚恋纠纷","劳资纠纷","单纯经济纠纷","未发现诈骗","未查实诈骗"]},
    {"category":"涉黄",
     "context_targets":["卖淫","嫖娼","色情服务","有偿陪侍","淫秽物品"],
     "structured":["涉黄","卖淫","嫖娼","淫秽物品","色情服务","组织卖淫","淫秽色情","传播淫秽"],
     "keywords":["涉黄","卖淫","嫖娼","色情服务","淫秽物品","组织卖淫","淫秽视频","色情交易","招嫖卡片","招嫖广告","传播淫秽"],
     "all_of":[[["有偿","收费"],["性服务","陪侍","性交易"]],[["酒店","宾馆","会所"],["招嫖","有偿性服务"]]],
     "exclude":["涉黄谣言","反黄宣传"]},
    {"category":"涉赌",
     "context_targets":["涉赌","赌博","赌资","赌场","赌局","赌钱"],
     "structured":["涉赌","赌博","开设赌场","赌资","赌具"],
     "keywords":["涉赌","赌博","聚众赌博","开设赌场","网络赌场","赌资","网络赌博","出资下注","地下赌场"],
     "all_of":[[["麻将","棋牌","扑克"],["赌资","筹码","赌钱"]],[["赌局","下注","押注"],["输钱","赢钱","输赢","赌资","押钱"]],
       [["赌博网站","网赌","网络平台"],["充值","下注"],["赌钱","赌博","输赢","下注"]]],
     "exclude":["棋牌娱乐无赌博","无赌资","未见赌资","未见赌博","无赌博行为"]},
    {"category":"食药环",
     "context_targets":["排放","外排","排污","污水","废水","危险废物","倾倒","污染","食品安全","食物中毒","假药","食品添加剂","超标","铝残留"],
     "structured":["食药环","食品药品","食品安全","危害食品安全","药品安全","环境污染","污染环境","制售假药","制售劣药","销售假药","制售有毒有害食品","不符合安全标准的食品","非法排污"],
     "keywords":["食药环","有毒有害食品","有毒、有害食品","假冒药品","制售假药","销售假药","制售劣药","非法排污","倾倒危险废物","非法处置危险废物","工业废水直排"],
     "all_of":[
       [["过期","变质","非法添加","违法添加","有害物质","有毒物质","地沟油"],["食品","饮料","肉制品","食用","减肥食品","包子","油条","糕点"],["销售","制作","生产","流通","加工"]],
       [["病死","病害"],["猪肉","牲畜","肉类"],["销售","加工","收购"]],
       [["药品","假药","劣药"],["伪造","制假","假药","劣药","无许可","无证"],["销售","兜售","制售","推销","生产"]],
       [["认定为假药","按假药论处","鉴定为假药","应按假药","检出不含标示成分","冒用药品批准文号"],["销售","贩卖","卖出","出售","待售","交易","售卖"]],
       [["假冒","仿冒","冒用药品名称"],["药品","药名","哮喘药","处方药","止咳药","止痛药","降压药","药膏","药片","胶囊"],["配制","分装","贴标","调配","自行生产"],["销售","贩卖","售卖","出售","待售"]],
       [["工业盐","工业用盐","化工用盐"],["当作食用盐","冒充食用盐","作为食用盐","用于做饭","用于烹饪","做饭时添加","给他人食用","食用盐标准"]],
       [["食品","面制品","油条","包子","糕点","蛋糕","面团","食用"],["铝残留量","铝的残留量","铝残留含量","铝含量","铝的含量","含铝膨松剂","含铝泡打粉","明矾"],["超标","不符合","超标准","超量添加","超剂量","超过规定","超过国家","超出","违反限量","不得添加"]],
       [["白酒","米酒","饮用酒","食品","肉制品"],["甲醇中毒","食物中毒","检出禁用成分"],["销售","售卖","出售","经营","生产"]],
       [["危险废物","工业废水","有毒污水","污染物"],["污水","废水","直排","排放","倾倒"]],
       [["污水","废水"],["直排","偷排","非法排放"]],
       [["废液","废水","污水","废酸","废碱","废盐酸","废机油","电镀液"],["暗管","渗坑","渗井","未经处理","未建废水处理设施","未经环保设施处理","排放超标"],["排放","排入","排出","倾倒","排到","直排"]],
       [["危险废物","危废","废旧化工容器"],["未经许可","无资质","没有危废经营许可证","未取得危险废物经营许可证"],["非法收购","收购","处置","翻新","倾倒","贩卖"]],
       [["工业染料","工业色素","非食用染料"],["食品","糕点","肉制品","饮料"],["加工","腌制","配制","制售","制作","生产"],["对外销售","售卖","销售","出售"]],
       [["药品","处方药","特效药"],["抽检证实系假药","鉴定系假药","检验确认系假药","认定为假药"],["销售","售卖","网售","兜售"]]
     ],
     "review_all_of":[
       [["包子","油条","蛋糕","糕点","面团","面制品"],["铝残留量","铝的残留量","铝残留含量","铝的残留含量","铝含量","铝的含量"],["检测","抽样","送检","检验"]],
       [["药品","药店","处方药","进口药"],["无国家食品药品监督管理局相关批准文号","无批准文号","无相关批准文号","批号查无","无进口药品注册证书","未获批进口"],["销售","贩卖","待售","出售"]],
       [["河砂","河沙","砂石料","河道","采矿","稀土"],["未取得采矿许可证","未取得合法采砂手续","无证开采","未经许可","未取得《河道采矿许可证》"],["非法开采","擅自采挖","擅自采矿","非法采矿","非法采砂","采挖河砂"]]
     ],
     "exclude":["已排除污染","已排除食物中毒","食品安全宣传","药品安全宣传","环境宣传教育"],
     "outcome_exclude":["检验结果合格","抽检结果合格","排放检测达标","未发现污染","未发现食品安全问题","查实为正常食用盐","检测符合国家标准","属于普通消费纠纷"]}
  ]
}
)json");
}

using Strings=std::vector<std::string>;
using Combinations=std::vector<std::vector<Strings>>;
struct Rule {
    std::string category;
    Strings structured,keywords,exclude,outcomeExclude;
    Combinations allOf,reviewAllOf;
    std::vector<std::wstring> contextTargets;
};
struct Rules {
    std::vector<Rule> classifiers;
    Strings review,outcomeReview;
};

Strings literals(const Json& input) {
    if(!input.is_array() || input.size()>512) throw std::invalid_argument("警情规则的关键词必须为数组。");
    auto values=input.get<Strings>();
    for(const auto& value:values) if(trimmed(value).empty() || value.size()>240 || value!=trimmed(value)) {
        throw std::invalid_argument("警情规则含空白或过长的关键词。");
    }
    return values;
}

Combinations combinations(const Json& groups) {
    if(!groups.is_array() || groups.size()>512) throw std::invalid_argument("警情组合词必须是数组。");
    Combinations result;
    for(const auto& group:groups) {
        if(!group.is_array() || group.empty() || group.size()>32) {
            throw std::invalid_argument("每个警情组合词组必须包含 1 至 32 项。");
        }
        std::vector<Strings> clauses;
        for(const auto& term:group) {
            auto options=literals(term.is_string()?Json::array({term}):term);
            if(options.empty()) throw std::invalid_argument("警情组合词的同义词列表不能留空。");
            clauses.push_back(std::move(options));
        }
        result.push_back(std::move(clauses));
    }
    return result;
}

Rules parseRules(const Json& input) {
    if(input.value("version",0)!=1 || !input.contains("rules") || !input["rules"].is_array() ||
       input["rules"].empty() || input["rules"].size()>64) {
        throw std::invalid_argument("规则 JSON 需要 version 1 和非空的 rules 数组。");
    }
    Rules result;
    result.review=literals(input.value("review_keywords",Json::array()));
    result.outcomeReview=literals(input.value("outcome_review_keywords",Json::array()));
    std::set<std::string> names;
    for(const auto& item:input["rules"]) {
        Rule rule;
        rule.category=trimmed(item.at("category").get<std::string>());
        if(rule.category.empty() || !names.insert(rule.category).second) throw std::invalid_argument("警情类别必须唯一且非空。");
        rule.structured=literals(item.value("structured",Json::array()));
        rule.keywords=literals(item.value("keywords",Json::array()));
        rule.exclude=literals(item.value("exclude",Json::array()));
        rule.outcomeExclude=literals(item.value("outcome_exclude",Json::array()));
        rule.allOf=combinations(item.value("all_of",Json::array()));
        rule.reviewAllOf=combinations(item.value("review_all_of",Json::array()));
        for(const auto& term:literals(item.value("context_targets",Json::array()))) rule.contextTargets.push_back(wide(term));
        if(rule.structured.empty() && rule.keywords.empty() && rule.allOf.empty() && rule.reviewAllOf.empty()) {
            throw std::invalid_argument("每个警情类别至少需要一条纳入或待复核规则。");
        }
        result.classifiers.push_back(std::move(rule));
    }
    return result;
}

std::string join(const Strings& values,std::string_view delimiter) {
    std::string result;
    for(const auto& value:values) {
        if(!result.empty()) result+=delimiter;
        result+=value;
    }
    return result;
}

std::optional<std::string> firstMatch(std::string_view text,const Strings& keywords) {
    const std::string* longest=nullptr;
    for(const auto& word:keywords) {
        if((!longest || word.size()>longest->size()) && text.find(word)!=std::string_view::npos) longest=&word;
    }
    return longest?std::optional<std::string>(*longest):std::nullopt;
}

std::optional<std::string> matchedCombination(std::string_view details,const Combinations& combinations) {
    for(const auto& group:combinations) {
        Strings matched;
        for(const auto& alternatives:group) {
            if(const auto word=firstMatch(details,alternatives)) matched.push_back(*word);
            else break;
        }
        if(matched.size()==group.size()) return join(matched,"＋");
    }
    return {};
}

std::wstring_view nearbyClause(std::wstring_view text,size_t start,bool acrossComma=false) {
    constexpr size_t maxLength=32;
    const auto input=text.substr(start,maxLength);
    return input.substr(0,input.find_first_of(acrossComma?L"。；！？;!?\r\n":L"，。；：！？,;!?\r\n"));
}

bool mentionsTarget(std::wstring_view clause,const std::vector<std::wstring>& targets) {
    return std::ranges::any_of(targets,[&](const std::wstring& word){return clause.find(word)!=std::wstring_view::npos;});
}

bool confirmedAfter(std::wstring_view text,size_t start,const std::vector<std::wstring>& targets) {
    constexpr std::array cues{L"确有",L"确认存在",L"确认有",L"查实存在",L"查实有人",L"查明存在",L"经复查发现",
        L"经核查发现",L"经核实发现",L"经查证实",L"现场查获",L"监控确认"};
    for(const std::wstring_view cue:cues) {
        for(auto pos=text.find(cue,start);pos!=std::wstring_view::npos;pos=text.find(cue,pos+cue.size())) {
            const auto before=text.substr(0,pos);
            if(before.ends_with(L"未") || before.ends_with(L"没有") || before.ends_with(L"无法") ||
               before.ends_with(L"不能") || before.ends_with(L"尚未")) continue;
            const auto clause=nearbyClause(text,pos+cue.size());
            if(clause.find(L"并无")!=std::wstring_view::npos || clause.find(L"没有")!=std::wstring_view::npos ||
               clause.find(L"不存在")!=std::wstring_view::npos || clause.find(L"未发现")!=std::wstring_view::npos ||
               clause.find(L"是否")!=std::wstring_view::npos) continue;
            if(mentionsTarget(clause,targets)) return true;
        }
    }
    return false;
}

bool canBeClarified(std::wstring_view word) {
    return word.find(L"未发现")!=std::wstring_view::npos || word.starts_with(L"未见") ||
           word.starts_with(L"没有") || word.starts_with(L"并未") || word.starts_with(L"无") ||
           word.find(L"不属实")!=std::wstring_view::npos || word.find(L"误会")!=std::wstring_view::npos;
}

std::optional<std::string> relevantReview(std::string_view input,const Strings& keywords,
        const std::vector<std::wstring>& targets,bool laterOutcomeConfirmed) {
    const auto word=firstMatch(input,keywords);
    if(!word || targets.empty()) return word;
    const auto w=wide(*word);
    if(!canBeClarified(w)) return word;
    if(laterOutcomeConfirmed) return {};
    const auto text=wide(std::string(input));
    const auto pos=text.rfind(w);
    return confirmedAfter(text,pos+w.size(),targets)?std::nullopt:word;
}

std::optional<std::string> contextualReview(std::string_view input,const std::vector<std::wstring>& targets,
                                             bool laterOutcomeConfirmed) {
    if(input.empty() || targets.empty() || laterOutcomeConfirmed) return {};
    const auto text=wide(std::string(input));
    constexpr std::array denials{L"未发现",L"未见",L"未发生",L"没有",L"并无",L"并未",L"不存在",
        L"查无",L"未检出",L"未证实",L"财物完好",L"未遭遇",L"否认"};
    constexpr std::array uncertainties{L"尚不能确定",L"无法确定",L"不能确定",L"未能确定",L"尚未确定",
        L"未明确",L"未说明",L"没有说明",L"尚不清楚",L"仍待核实"};
    const auto match=[&](const auto& cues,bool uncertain)->std::optional<std::string> {
        for(const std::wstring_view cue:cues) {
            for(auto pos=text.find(cue);pos!=std::wstring::npos;pos=text.find(cue,pos+cue.size())) {
                const auto clause=nearbyClause(text,pos,uncertain);
                if(uncertain && clause.find(L"是否")==std::wstring::npos && clause.find(L"有无")==std::wstring::npos &&
                   clause.find(L"有没有")==std::wstring::npos) continue;
                if(mentionsTarget(clause,targets) && !confirmedAfter(text,pos+clause.size(),targets)) return utf8(clause);
            }
        }
        return {};
    };
    if(const auto uncertain=match(uncertainties,true)) return uncertain;
    return match(denials,false);
}

std::string displayTime(std::string_view cell,bool date1904) {
    double serial{};
    const auto result=std::from_chars(cell.data(),cell.data()+cell.size(),serial);
    if(result.ec!=std::errc{} || result.ptr!=cell.data()+cell.size() || !std::isfinite(serial) ||
       serial<1 || serial>2958465.) return std::string(cell);
    using namespace std::chrono;
    const auto epoch=date1904?sys_days{1904y/January/1}:serial<60.?sys_days{1899y/December/31}:sys_days{1899y/December/30};
    const auto full=epoch+seconds{static_cast<int64_t>(std::llround(serial*86400.))};
    const auto day=floor<days>(full);
    const year_month_day date(day);
    const hh_mm_ss time(full-day);
    std::ostringstream text;
    text<<std::setfill('0')<<std::setw(4)<<int(date.year())<<'-'<<std::setw(2)<<unsigned(date.month())<<'-'
        <<std::setw(2)<<unsigned(date.day())<<' '<<std::setw(2)<<time.hours().count()<<':'
        <<std::setw(2)<<time.minutes().count()<<':'<<std::setw(2)<<time.seconds().count();
    return text.str();
}

struct Classification {
    Strings categories,positive,negative;
    bool review{};
};

Classification classify(const Worksheet& sheet,const Row& row,const Rules& rules) {
    const auto field=[&](const char* name) {
        const auto found=sheet.headers.find(name);
        return found==sheet.headers.end()?std::string_view{}:valueAt(row,found->second);
    };
    const auto details=field("事件详情");
    const std::array sources{"报警子类","报警细类","报警类型","报警类别"};
    const std::array outcomeFields{"反馈内容","处理结果"};
    const std::array narrativeFields{"事件详情","反馈内容","处理结果","报警子类","报警细类","报警类型"};
    struct Hit {std::string category;Strings positive,negative;size_t priority;bool uncertain;};
    std::vector<Hit> hits;
    for(const auto& rule:rules.classifiers) {
        Hit hit{rule.category,{},{},sources.size(),false};
        for(size_t i=0;i<sources.size();++i) {
            if(const auto term=firstMatch(field(sources[i]),rule.structured)) {
                hit.priority=i;
                hit.positive.push_back(std::string(sources[i])+"「"+*term+"」");
                break;
            }
        }
        if(const auto term=firstMatch(details,rule.keywords)) {
            hit.positive.push_back("事件详情「"+*term+"」");
        } else if(const auto combination=matchedCombination(details,rule.allOf)) {
            hit.positive.push_back("事件详情组合「"+*combination+"」");
        } else if(hit.positive.empty()) {
            if(const auto weak=matchedCombination(details,rule.reviewAllOf)) {
                hit.positive.push_back("事件详情待核实组合「"+*weak+"」");
                hit.uncertain=true;
            }
        }
        if(hit.positive.empty()) continue;
        const bool outcomeConfirmed=!rule.contextTargets.empty() && std::ranges::any_of(outcomeFields,
            [&](const char* name){return confirmedAfter(wide(std::string(field(name))),0,rule.contextTargets);});
        for(const auto* name:narrativeFields) {
            const auto value=field(name);
            const bool clarified=outcomeConfirmed && std::string_view(name)=="事件详情";
            const auto explicitReview=relevantReview(value,rule.exclude,rule.contextTargets,clarified);
            const auto generalReview=relevantReview(value,rules.review,rule.contextTargets,clarified);
            const auto context=contextualReview(value,rule.contextTargets,clarified);
            const auto term=explicitReview?explicitReview:generalReview?generalReview:context;
            if(term) hit.negative.push_back(std::string(name)+"「"+*term+"」");
        }
        for(const auto* name:outcomeFields) {
            const auto value=field(name);
            const auto specific=firstMatch(value,rule.outcomeExclude);
            const auto term=specific?specific:firstMatch(value,rules.outcomeReview);
            if(term) hit.negative.push_back(std::string(name)+"「"+*term+"」");
        }
        hits.push_back(std::move(hit));
    }
    std::stable_sort(hits.begin(),hits.end(),[](const Hit& a,const Hit& b){return a.priority<b.priority;});
    Classification result;
    for(const auto& hit:hits) {
        result.categories.push_back(hit.category);
        for(const auto& reason:hit.positive) result.positive.push_back(hit.category+" · "+reason);
        for(const auto& reason:hit.negative) result.negative.push_back(hit.category+" · "+reason);
        result.review=result.review || hit.uncertain;
    }
    result.review=result.review || !result.negative.empty() || result.categories.size()>1;
    return result;
}

std::string xmlEscape(std::string_view input) {
    std::string text;
    for(const auto ch:input) {
        switch(ch) {
        case '&': text+="&amp;"; break;
        case '<': text+="&lt;"; break;
        case '>': text+="&gt;"; break;
        case '"': text+="&quot;"; break;
        case '\r': text+="&#13;"; break;
        default: text+=ch; break;
        }
    }
    return text;
}

void appendCell(std::string& xml,size_t row,const Cell& cell,bool keepStyle) {
    xml+="<c r=\""+columnName(cell.column)+std::to_string(row)+"\"";
    if(keepStyle && !cell.style.empty()) xml+=" s=\""+xmlEscape(cell.style)+"\"";
    if(cell.type=="inlineStr") xml+=" t=\"inlineStr\"><is><t xml:space=\"preserve\">"+xmlEscape(cell.value)+"</t></is></c>";
    else {
        if(!cell.type.empty()) xml+=" t=\""+xmlEscape(cell.type)+"\"";
        if(cell.value.empty()) xml+="/>";
        else xml+="><v>"+xmlEscape(cell.value)+"</v></c>";
    }
}

ComPtr<IOpcPart> addPart(IOpcFactory* factory,IOpcPartSet* parts,const wchar_t* name,const wchar_t* contentType,
                         const std::string& xml) {
    ComPtr<IOpcPartUri> uri;
    ComPtr<IOpcPart> part;
    check(factory->CreatePartUri(name,&uri),"无法创建 Excel 部件地址。");
    check(parts->CreatePart(uri.Get(),contentType,OPC_COMPRESSION_NORMAL,&part),"无法创建 Excel 部件。");
    ComPtr<IStream> stream;
    check(part->GetContentStream(&stream),"无法写入筛选 Excel。");
    ULONG written{};
    if(xml.size()>static_cast<size_t>(ULONG_MAX)) throw std::invalid_argument("导出 Excel 超出可写入大小。");
    check(stream->Write(xml.data(),static_cast<ULONG>(xml.size()),&written),"无法写入筛选 Excel。");
    if(written!=xml.size()) throw std::runtime_error("筛选 Excel 写入不完整。");
    return part;
}

void relate(IOpcRelationshipSet* relationships,IOpcPart* part,const wchar_t* id,const wchar_t* kind) {
    ComPtr<IOpcPartUri> uri;
    check(part->GetName(&uri),"无法读取 Excel 部件地址。");
    ComPtr<IOpcRelationship> result;
    const auto type=std::wstring(relationshipPrefix)+kind;
    check(relationships->CreateRelationship(id,type.c_str(),uri.Get(),OPC_URI_TARGET_MODE_INTERNAL,&result),
          "无法写入 Excel 工作簿关系。");
}

struct Selection {
    const Row* row{};
    Classification label;
};

void exportWorkbook(IOpcFactory* factory,const Worksheet& source,const std::vector<Selection>& selected,
                    const fs::path& target,const std::atomic<bool>& cancel) {
    if(_wcsicmp(target.extension().c_str(),L".xlsx")!=0) throw std::invalid_argument("导出文件扩展名必须是 .xlsx。");
    if(source.columns+3>16384) throw std::invalid_argument("Excel 列数超过导出上限。");
    const bool styled=!source.styleXml.empty();
    const auto addRow=[&](std::string& xml,size_t index,const Row& original,const Strings& labels) {
        xml+="<row r=\""+std::to_string(index)+"\">";
        for(const auto& cell:original.values) appendCell(xml,index,cell,styled);
        for(size_t i=0;i<labels.size();++i) appendCell(xml,index,{source.columns+i,labels[i],"inlineStr",{}},false);
        xml+="</row>";
    };
    std::string sheet="<?xml version=\"1.0\" encoding=\"UTF-8\"?><worksheet xmlns=\""+std::string(sheetNamespace)+"\"><sheetData>";
    addRow(sheet,1,source.rows.front(),{"命中类别","筛选依据","筛选状态"});
    for(size_t i=0;i<selected.size();++i) {
        if(cancel) return;
        const auto& entry=selected[i];
        auto reason=join(entry.label.positive,"；");
        if(!entry.label.negative.empty()) reason+="。冲突证据："+join(entry.label.negative,"；");
        addRow(sheet,i+2,*entry.row,{join(entry.label.categories,"、"),reason,entry.label.review?"待复核":"命中"});
    }
    sheet+="</sheetData></worksheet>";
    const std::string workbook="<?xml version=\"1.0\" encoding=\"UTF-8\"?><workbook xmlns=\""+
        std::string(sheetNamespace)+"\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"+
        (source.date1904?"<workbookPr date1904=\"1\"/>":"")+
        "<sheets><sheet name=\"警情筛选\" sheetId=\"1\" r:id=\"rId1\"/></sheets></workbook>";
    ComPtr<IOpcPackage> package;
    ComPtr<IOpcPartSet> parts;
    ComPtr<IOpcRelationshipSet> root;
    check(factory->CreatePackage(&package),"无法创建筛选 Excel。");
    check(package->GetPartSet(&parts),"无法创建筛选 Excel 部件。");
    check(package->GetRelationshipSet(&root),"无法创建筛选 Excel 关系。");
    const auto book=addPart(factory,parts.Get(),L"/xl/workbook.xml",
        L"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",workbook);
    relate(root.Get(),book.Get(),L"rId1",L"officeDocument");
    ComPtr<IOpcRelationshipSet> bookRelationships;
    check(book->GetRelationshipSet(&bookRelationships),"无法创建工作表关系。");
    const auto page=addPart(factory,parts.Get(),L"/xl/worksheets/sheet1.xml",
        L"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml",sheet);
    relate(bookRelationships.Get(),page.Get(),L"rId1",L"worksheet");
    if(styled) {
        const auto style=addPart(factory,parts.Get(),L"/xl/styles.xml",
            L"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml",source.styleXml);
        relate(bookRelationships.Get(),style.Get(),L"rId2",L"styles");
    }
    const auto temporary=target.parent_path()/(L"."+wide(newId())+L".tmp");
    struct TempFile {
        fs::path path;
        ~TempFile() {std::error_code error;fs::remove(path,error);}
    } cleanup{temporary};
    ComPtr<IStream> file;
    check(factory->CreateStreamOnFile(temporary.c_str(),OPC_STREAM_IO_WRITE,nullptr,FILE_ATTRIBUTE_NORMAL,&file),
          "无法创建 Excel 导出文件。");
    check(factory->WritePackageToStream(package.Get(),OPC_WRITE_DEFAULT,file.Get()),"无法写入 Excel 导出文件。");
    file.Reset();
    if(!cancel && !MoveFileExW(temporary.c_str(),target.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)) {
        throw std::runtime_error("无法保存筛选 Excel，请关闭占用该文件的程序。");
    }
}

std::string fingerprint(const fs::path& source) {
    return std::to_string(fs::file_size(source))+":"+std::to_string(fs::last_write_time(source).time_since_epoch().count());
}

}

Json incidentOperation(const std::string& action,const Json& request,const std::atomic<bool>& cancel) {
    if(action=="incident.rules") {
        auto rules=defaultRules();
        if(request.contains("output")) writeJson(localPath(textPath(request.at("output").get<std::string>())),rules);
        return rules;
    }
    const auto source=localPath(textPath(request.at("source").get<std::string>()));
    const auto stamp=fingerprint(source);
    if(request.contains("source_stamp") && request.at("source_stamp")!=stamp) {
        throw std::invalid_argument("源警情表已变化，请重新筛选和复核。");
    }
    const auto payload=request.contains("rules")?request.at("rules"):request.contains("rules_path")?
        readJson(localPath(textPath(request.at("rules_path").get<std::string>())),1024*1024):defaultRules();
    const auto rules=parseRules(payload);
    const Apartment apartment;
    ComPtr<IOpcFactory> factory;
    check(CoCreateInstance(__uuidof(OpcFactory),nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&factory)),
          "无法初始化 Windows Excel 工作簿组件。");
    const auto sheet=readWorkbook(factory.Get(),source,cancel);
    if(cancel) return {{"cancelled",true}};
    if(fingerprint(source)!=stamp) throw std::runtime_error("读取期间警情表已变化，请重新打开。");
    const auto field=[&](const Row& row,const char* name) {
        const auto column=sheet.headers.find(name);
        return column==sheet.headers.end()?std::string_view{}:valueAt(row,column->second);
    };
    std::vector<Selection> selected;
    size_t matched=0,review=0;
    for(size_t i=1;i<sheet.rows.size();++i) {
        if(cancel) return {{"cancelled",true}};
        const auto& row=sheet.rows[i];
        auto result=classify(sheet,row,rules);
        if(result.categories.empty()) continue;
        if(result.review) ++review;
        else ++matched;
        selected.push_back({&row,std::move(result)});
    }
    const auto total=sheet.rows.size()-1;
    if(action=="incident.export") {
        const auto target=localPath(textPath(request.at("output").get<std::string>()));
        std::error_code error;
        if(_wcsicmp(source.c_str(),target.c_str())==0 || fs::equivalent(source,target,error)) {
            throw std::invalid_argument("请选择不同于警情原表的输出文件。");
        }
        if(!request.value("include_review",true)) std::erase_if(selected,[](const auto& entry){return entry.label.review;});
        exportWorkbook(factory.Get(),sheet,selected,target,cancel);
        if(cancel) return {{"cancelled",true}};
        return {{"output",pathText(target)},{"total",total},{"exported",selected.size()},{"matched",matched},{"review",review}};
    }
    Json hits=Json::array();
    std::string text="已读取 "+std::to_string(total)+" 条 · 命中 "+std::to_string(matched)+" 条 · 待复核 "+std::to_string(review)+" 条\r\n";
    for(const auto& entry:selected) {
        const auto& row=*entry.row;
        const auto& label=entry.label;
        const auto time=displayTime(field(row,"报警时间"),sheet.date1904);
        const auto status=label.review?"待复核":"命中";
        hits.push_back({{"row",row.original},{"status",status},{"categories",label.categories},
            {"evidence",label.positive},{"conflicts",label.negative},{"time",time},{"address",field(row,"事发地址")},
            {"details",field(row,"事件详情")},{"feedback",field(row,"反馈内容")}});
        text+="\r\n第 "+std::to_string(row.original)+" 行 · "+status+" · "+join(label.categories,"、")+"\r\n";
        text+="时间："+time+"　地址："+std::string(field(row,"事发地址"))+"\r\n";
        text+="事件详情："+std::string(field(row,"事件详情"))+"\r\n";
        text+="命中依据："+join(label.positive,"；")+"\r\n";
        if(!label.negative.empty()) text+="冲突证据："+join(label.negative,"；")+"\r\n";
    }
    return {{"total",total},{"matched",matched},{"review",review},{"hits",std::move(hits)},{"review_text",text},
        {"export_request",{{"source",pathText(source)},{"source_stamp",stamp},{"rules",payload}}}};
}

}
