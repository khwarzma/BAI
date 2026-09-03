#include "tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
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
    size_t offset = 0;
    while (offset < text.size()) {
        const size_t start = offset;
        uint32_t code_point = 0;
        if (!decode_utf8(text, offset, code_point)) {
            continue;
        }
        if (code_point <= 0x7f && std::isalnum(static_cast<unsigned char>(code_point))) {
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(code_point))));
        } else if (code_point == ' ' || code_point == '\t' || code_point == '\n' || code_point == '\r') {
            if (!out.empty() && out.back() != ' ') {
                out.push_back(' ');
            }
        } else if (code_point == '-' || code_point == '_' || code_point == '.') {
            out.append(text, start, offset - start);
        } else if (code_point >= 0x80 && code_point != 0x200b) {
            out.append(text, start, offset - start);
        }
    }
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

bool BaiTokenizer::decode_utf8(
        const std::string& text,
        size_t& offset,
        uint32_t& code_point
    ) {
        const auto byte = [&](size_t index) {
            return static_cast<unsigned char>(text[index]);
        };
        const size_t remaining = text.size() - offset;
        const unsigned char first = byte(offset);
        size_t width = 0;
        uint32_t value = 0;
        if (first <= 0x7f) {
            width = 1;
            value = first;
        } else if (first >= 0xc2 && first <= 0xdf) {
            width = 2;
            value = first & 0x1f;
        } else if (first >= 0xe0 && first <= 0xef) {
            width = 3;
            value = first & 0x0f;
        } else if (first >= 0xf0 && first <= 0xf4) {
            width = 4;
            value = first & 0x07;
        } else {
            ++offset;
            return false;
        }
        if (remaining < width) {
            ++offset;
            return false;
        }
        for (size_t i = 1; i < width; ++i) {
            const unsigned char continuation = byte(offset + i);
            if ((continuation & 0xc0) != 0x80) {
                ++offset;
                return false;
            }
            value = (value << 6) | (continuation & 0x3f);
        }
        if ((width == 2 && value < 0x80) ||
            (width == 3 && value < 0x800) ||
            (width == 4 && value < 0x10000) ||
            (value > 0x10ffff) ||
            (value >= 0xd800 && value <= 0xdfff)) {
            ++offset;
            return false;
        }
        offset += width;
        code_point = value;
        return true;
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

    std::vector<size_t> boundaries{0};
    size_t offset = 0;
    while (offset < word.size()) {
        uint32_t code_point = 0;
        if (!decode_utf8(word, offset, code_point)) {
            continue;
        }
        boundaries.push_back(offset);
    }

    size_t remaining_index = 0;
    while (remaining_index + 1 < boundaries.size()) {
        bool matched = false;
        const size_t remaining_code_points = boundaries.size() - 1 - remaining_index;
        for (size_t len = remaining_code_points; len > 0; --len) {
            const size_t end = boundaries[remaining_index + len];
            std::string candidate = word.substr(boundaries[remaining_index], end - boundaries[remaining_index]);
            if (len != remaining_code_points) {
                candidate = "##" + candidate;
            }
            auto it = vocab_.find(candidate);
            if (it != vocab_.end()) {
                result.push_back(it->second);
                remaining_index += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            result.push_back(unk_token_id_);
            ++remaining_index;
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
    if (vocab_.size() != 50257) {
        return std::unexpected(
            "Vocabulary must contain 50257 entries; loaded " +
            std::to_string(vocab_.size()) + " from " + vocab_path
        );
    }
    return {};
}

}  // namespace bai
