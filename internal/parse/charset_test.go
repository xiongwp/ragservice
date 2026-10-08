package parse

import (
	"strings"
	"testing"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/charmap"
	"golang.org/x/text/encoding/japanese"
	"golang.org/x/text/encoding/korean"
	"golang.org/x/text/encoding/simplifiedchinese"
	"golang.org/x/text/encoding/traditionalchinese"
	"golang.org/x/text/encoding/unicode"
)

func encode(t *testing.T, enc encoding.Encoding, s string) []byte {
	t.Helper()
	b, err := enc.NewEncoder().Bytes([]byte(s))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

var gb = simplifiedchinese.GB18030

// Paragraph-length samples, as found in real files.
var samples = []struct {
	name, want string
	enc        encoding.Encoding
	text       string
}{
	{"chinese GBK", "gb18030", simplifiedchinese.GBK,
		"员工每年享有带薪年假。工龄满一年不满十年的，年假为五天；满十年不满二十年的，年假为十天。报销流程：填写报销单并附上发票，部门经理审批，财务复核后五个工作日内打款。"},
	{"traditional Big5", "big5", traditionalchinese.Big5,
		"員工每年享有帶薪年假。工齡滿一年不滿十年的，年假為五天；滿十年不滿二十年的，年假為十天。報銷流程：填寫報銷單並附上發票。"},
	{"japanese Shift_JIS", "shift_jis", japanese.ShiftJIS,
		"従業員は毎年有給休暇を取得できます。勤続一年以上十年未満の場合、年次休暇は五日間です。経費精算の流れ：申請書に領収書を添付し、部長が承認します。"},
	{"japanese EUC-JP", "euc-jp", japanese.EUCJP,
		"従業員は毎年有給休暇を取得できます。勤続一年以上十年未満の場合、年次休暇は五日間です。経費精算の流れ：申請書に領収書を添付し、部長が承認します。"},
	{"korean EUC-KR", "euc-kr", korean.EUCKR,
		"직원은 매년 유급 휴가를 받을 수 있습니다. 근속 1년 이상 10년 미만인 경우 연차 휴가는 5일입니다. 경비 정산 절차: 영수증을 첨부하여 신청서를 제출합니다."},
	{"french windows-1252", "windows-1252", charmap.Windows1252,
		"Les employés bénéficient chaque année de congés payés. Après dix ans d'ancienneté, la durée des congés est portée à dix jours ouvrés. Remboursement : joindre les reçus."},
	{"russian windows-1251", "windows-1251", charmap.Windows1251,
		"Сотрудники ежегодно получают оплачиваемый отпуск. При стаже от одного до десяти лет продолжительность отпуска составляет пять дней. Возмещение расходов: приложите чеки."},
	{"greek ISO-8859-7", "iso-8859-7", charmap.ISO8859_7,
		"Οι εργαζόμενοι δικαιούνται κάθε χρόνο άδεια μετ' αποδοχών. Για προϋπηρεσία από ένα έως δέκα έτη η άδεια είναι πέντε ημέρες."},
}

func TestDecodeTextDetectsLegacyEncodings(t *testing.T) {
	for _, s := range samples {
		got, name := decodeText(encode(t, s.enc, s.text), gb, "gb18030")
		if got != s.text || name != s.want {
			t.Errorf("%s: decoded as %s, text ok=%v", s.name, name, got == s.text)
		}
	}
}

func TestDecodeTextUnicode(t *testing.T) {
	text := "多语言 mixed テキスト 한국어"
	cases := map[string][]byte{
		"utf-8":          []byte(text),
		"utf-8 with BOM": append([]byte{0xEF, 0xBB, 0xBF}, text...),
		"utf-16le BOM":   encode(t, unicode.UTF16(unicode.LittleEndian, unicode.UseBOM), text),
		"utf-16be BOM":   encode(t, unicode.UTF16(unicode.BigEndian, unicode.UseBOM), text),
	}
	for name, data := range cases {
		if got, _ := decodeText(data, gb, "gb18030"); got != text {
			t.Errorf("%s: got %q", name, got)
		}
	}
}

// Too short for statistics: the configured fallback decides.
func TestDecodeTextFallback(t *testing.T) {
	short := "年假为十天。"
	if got, name := decodeText(encode(t, simplifiedchinese.GBK, short), gb, "gb18030"); got != short || name != "gb18030" {
		t.Errorf("GBK short text: %q via %s", got, name)
	}
	ja := "有給休暇。"
	sjis, sjisName, ok := FallbackEncoding("Shift_JIS")
	if !ok {
		t.Fatal("shift_jis label not found")
	}
	if got, _ := decodeText(encode(t, japanese.ShiftJIS, ja), sjis, sjisName); got != ja {
		t.Errorf("Shift_JIS short text with Japanese fallback: %q", got)
	}
}

func TestDecodeHTML(t *testing.T) {
	ja := "従業員は毎年有給休暇を取得できます。"
	page := func(meta, body string) string {
		return "<html><head>" + meta + "<title>t</title></head><body><p>" + body + "</p></body></html>"
	}
	// A declared charset wins, even for a body too short to detect.
	declared := encode(t, japanese.ShiftJIS, page(`<meta charset="shift_jis">`, ja))
	if got, name := decodeHTML(declared, gb, "gb18030"); !strings.Contains(got, ja) || name != "shift_jis" {
		t.Errorf("meta charset: %s %q", name, got)
	}
	httpEquiv := encode(t, simplifiedchinese.GBK, page(`<meta http-equiv="Content-Type" content="text/html; charset=gbk">`, "年假"))
	if got, _ := decodeHTML(httpEquiv, gb, "gb18030"); !strings.Contains(got, "年假") {
		t.Errorf("http-equiv charset: %q", got)
	}
	// Nothing declared: detected, not defaulted to windows-1252.
	ko := samples[4].text
	undeclared := encode(t, korean.EUCKR, page("", ko))
	if got, name := decodeHTML(undeclared, gb, "gb18030"); !strings.Contains(got, ko) || name != "euc-kr" {
		t.Errorf("undeclared EUC-KR page: %s", name)
	}
	// A page that really declares Latin-1 keeps it.
	fr := "Remboursement : joindre les reçus."
	latin := encode(t, charmap.Windows1252, page(`<meta charset="iso-8859-1">`, fr))
	if got, _ := decodeHTML(latin, gb, "gb18030"); !strings.Contains(got, fr) {
		t.Errorf("declared latin-1: %q", got)
	}
}

func TestParseUsesDetection(t *testing.T) {
	p := New("", 0, 1)
	md, err := p.Parse(t.Context(), "x.txt", "txt", encode(t, japanese.ShiftJIS, samples[2].text))
	if err != nil || md != samples[2].text {
		t.Errorf("txt: %v %q", err, md)
	}
	if err := p.SetFallbackEncoding("no-such-encoding"); err == nil {
		t.Error("unknown fallback encoding accepted")
	}
}
