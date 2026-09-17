// nntc_json.h: the small JSON reader, shared by the encoder (the MATERIAL json it can be given instead of a list of
// images), the viewers and bc_check (the ASSET's own PREFIX_nntc.json). It sits in shared/ with the other files more
// than one program in this tree reads, which is where it went when a second viewer made src/ - the encoder's own
// sources - the wrong home for it.
// The parsing is sheredom's json.h (one header, public domain, https://github.com/sheredom/json.h); this file only lifts its
// linked-list DOM into the small JVal tree the callers walk (objects, arrays, numbers, strings, true / false / null).
#pragma once
#include <string>
#include <vector>
#include <cstdlib>
#include "json.h"

struct JVal {
    enum Kind { NUL, BOOL, NUM, STR, ARR, OBJ } kind = NUL;
    double num = 0; bool b = false; std::string str;
    std::vector<JVal> arr; std::vector<std::pair<std::string, JVal>> obj;
    const JVal* get(const char* key) const { for (auto& kv : obj) if (kv.first == key) return &kv.second; return nullptr; }
    double number(const char* key, double def = 0) const { const JVal* v = get(key); return v && v->kind == NUM ? v->num : def; }
    std::string string(const char* key) const { const JVal* v = get(key); return v && v->kind == STR ? v->str : std::string(); }
};
static JVal jval_from(const json_value_s* v) {
    JVal r;
    switch (v->type) {
    case json_type_string: { const json_string_s* s = (const json_string_s*)v->payload; r.kind = JVal::STR; r.str.assign(s->string, s->string_size); break; }
    case json_type_number: { const json_number_s* n = (const json_number_s*)v->payload; r.kind = JVal::NUM; r.num = strtod(std::string(n->number, n->number_size).c_str(), nullptr); break; }
    case json_type_object: { r.kind = JVal::OBJ; for (const json_object_element_s* e = ((const json_object_s*)v->payload)->start; e; e = e->next) r.obj.push_back({ std::string(e->name->string, e->name->string_size), jval_from(e->value) }); break; }
    case json_type_array: { r.kind = JVal::ARR; for (const json_array_element_s* e = ((const json_array_s*)v->payload)->start; e; e = e->next) r.arr.push_back(jval_from(e->value)); break; }
    case json_type_true: r.kind = JVal::BOOL; r.b = true; break;
    case json_type_false: r.kind = JVal::BOOL; break;
    default: break;
    }
    return r;
}
// What the parser refused, in words. A material JSON is written by hand, so "it did not parse" is not an answer: the
// caller wants to be told what was wrong and where, and the two things a hand-written file most often carries - a
// trailing comma and a byte-order mark - are indistinguishable from every other failure without this.
inline const char* json_error_name(int error) {
    switch (error) {
    case json_parse_error_expected_comma_or_closing_bracket: return "a comma or a closing bracket was expected";
    case json_parse_error_expected_colon: return "a colon between a name and its value was expected";
    case json_parse_error_expected_opening_quote: return "a string was expected to open with a quote";
    case json_parse_error_invalid_string_escape_sequence: return "a string carries an invalid escape sequence";
    case json_parse_error_invalid_number_format: return "a number is not in a valid format";
    case json_parse_error_invalid_value: return "a value is not valid JSON";
    case json_parse_error_premature_end_of_buffer: return "the file ends before the value is complete";
    case json_parse_error_invalid_string: return "a string is malformed";
    case json_parse_error_allocator_failed: return "the parser could not allocate";
    case json_parse_error_unexpected_trailing_characters: return "there is text after the value the file holds";
    case json_parse_error_recursion: return "the objects and arrays are nested too deeply";
    default: return "the text is not valid JSON";
    }
}

// The same shape the callers used: P.parse() gives the root, P.ok says whether the text parsed. A failure also carries
// WHERE it happened: json_parse_ex fills the line and the column of the offending byte, and the location flag is asked
// for so that the DOM the successful path walks carries its own positions too, should a caller ever want them.
struct JParser {
    const std::string& s; bool ok = true;
    size_t error_line = 0, error_column = 0;   // 1-based, as an editor counts them; 0 when nothing failed
    const char* error_text = "";
    explicit JParser(const std::string& src) : s(src) {}
    JVal parse() {
        json_parse_result_s res;
        res.error = 0; res.error_offset = 0; res.error_line_no = 0; res.error_row_no = 0;
        json_value_s* root = json_parse_ex(s.data(), s.size(), json_parse_flags_allow_location_information, nullptr,
                                           nullptr, &res);
        if (!root) {
            ok = false;
            error_line = res.error_line_no;
            error_column = res.error_row_no;
            error_text = json_error_name((int)res.error);
            return JVal();
        }
        JVal r = jval_from(root); free(root); return r;
    }
};
