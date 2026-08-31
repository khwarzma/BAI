#include "tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace bai {

BaiTokenizer::BaiTokenizer() = default;

std::expected<void, std::string> BaiTokenizer::initialize(const std::string& vocab_path) {
    return load_vocab_json(vocab_path);
}

std::expected<BaiTokenizer::TokenizedOutput, std::string> BaiTokenizer::encode(
    const std::string& text,
    size_t max_seq_len
) {
    if (!is_initialized()) {
        return std::unexpected("Tokenizer not initialized. Call initialize() first.");
    }
    if (text.empty()) {
        return std::unexpected("Input text is empty.");
    }
    if (max_seq_len < 3) {
        return std::unexpected("max_seq_len must be at least 3.");
    }

    const std::string normalized = normalize_text(text);
    const auto words = whitespace_tokenize(normalized);

    std::vector<int64_t> token_ids;
    token_ids.push_back(cls_token_id_);

    for (const auto& word : words) {
        if (token_ids.size() >= max_seq_len - 1) {
            break;
        }
        const auto word_tokens = tokenize_word(word);
        for (int64_t token_id : word_tokens) {
            if (token_ids.size() >= max_seq_len - 1) {
                break;
            }
            token_ids.push_back(token_id);
        }
    }

    token_ids.push_back(sep_token_id_);
    const size_t actual_length = token_ids.size();

    std::vector<int64_t> attention_mask(token_ids.size(), 1);
    while (token_ids.size() < max_seq_len) {
        token_ids.push_back(pad_token_id_);
        attention_mask.push_back(0);
    }

    return TokenizedOutput{std::move(token_ids), std::move(attention_mask), actual_length};
}

std::string BaiTokenizer::normalize_text(const std::string& text) const {
    std::string out;
    out.reserve(text.size());
    for (unsigned char ch : text) {
        if (std::isalnum(ch)) {
            out.push_back(static_cast<char>(std::tolower(ch)));
        } else if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
            if (!out.empty() && out.back() != ' ') {
                out.push_back(' ');
            }
        } else if (ch == '-' || ch == '_' || ch == '.') {
            out.push_back(static_cast<char>(ch));
        }
    }
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

std::vector<std::string> BaiTokenizer::whitespace_tokenize(const std::string& text) const {
    std::istringstream iss(text);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<int64_t> BaiTokenizer::tokenize_word(const std::string& word) const {
    std::vector<int64_t> result;
    if (word.empty()) {
        return result;
    }

    auto exact = vocab_.find(word);
    if (exact != vocab_.end()) {
        result.push_back(exact->second);
        return result;
    }

    std::string remaining = word;
    while (!remaining.empty()) {
        bool matched = false;
        for (size_t len = remaining.size(); len > 0; --len) {
            std::string candidate = remaining.substr(0, len);
            if (len != remaining.size()) {
                candidate = "##" + candidate;
            }
            auto it = vocab_.find(candidate);
            if (it != vocab_.end()) {
                result.push_back(it->second);
                remaining = remaining.substr(len);
                matched = true;
                break;
            }
        }
        if (!matched) {
            result.push_back(unk_token_id_);
            remaining = remaining.substr(1);
        }
    }
    return result;
}

std::expected<void, std::string> BaiTokenizer::load_vocab_json(const std::string& vocab_path) {
    std::ifstream in(vocab_path);
    if (!in.is_open()) {
        return std::unexpected("Failed to open vocabulary file: " + vocab_path);
    }

    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    vocab_.clear();
    reverse_vocab_.clear();

    std::string token_name;
    std::string number_str;
    size_t i = 0;
    while (i < content.size()) {
        const auto quote1 = content.find('"', i);
        if (quote1 == std::string::npos) {
            break;
        }
        const auto quote2 = content.find('"', quote1 + 1);
        if (quote2 == std::string::npos) {
            break;
        }
        token_name = content.substr(quote1 + 1, quote2 - quote1 - 1);
        auto colon = content.find(':', quote2 + 1);
        if (colon == std::string::npos) {
            break;
        }
        auto number_start = content.find_first_not_of(" \t\r\n", colon + 1);
        if (number_start == std::string::npos) {
            break;
        }
        auto number_end = content.find_first_of(",}\n\r", number_start);
        number_str = content.substr(number_start, number_end - number_start);
        constexpr auto trim = [](std::string s) {
            while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n')) {
                s.erase(s.begin());
            }
            while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) {
                s.pop_back();
            }
            return s;
        };
        const std::string clean = trim(number_str);
        const long long value = std::stoll(clean);
        vocab_[token_name] = static_cast<int64_t>(value);
        reverse_vocab_[static_cast<int64_t>(value)] = token_name;
        i = number_end == std::string::npos ? content.size() : number_end + 1;
    }

    if (vocab_.empty()) {
        return std::unexpected("Vocabulary file is empty or malformed: " + vocab_path);
    }
    if (vocab_.find("[PAD]") != vocab_.end()) pad_token_id_ = vocab_["[PAD]"];
    if (vocab_.find("[CLS]") != vocab_.end()) cls_token_id_ = vocab_["[CLS]"];
    if (vocab_.find("[SEP]") != vocab_.end()) sep_token_id_ = vocab_["[SEP]"];
    if (vocab_.find("[UNK]") != vocab_.end()) unk_token_id_ = vocab_["[UNK]"];
    return {};
}

}  // namespace bai
