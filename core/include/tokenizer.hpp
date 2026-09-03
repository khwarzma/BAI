
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <expected>
#include <cstdint>

namespace bai {

/**
 * BaiTokenizer - Encodes text to token IDs and attention masks.
 * 
 * Loads subword vocabulary and provides efficient text-to-tokens encoding
 * for BaiMicroEncoder inference. Handles:
 * - UTF-8 text normalization
 * - Subword tokenization via byte-pair encoding vocabulary
 * - Attention mask generation (1=token, 0=padding)
 * - Sequence length capping
 */
class BaiTokenizer {
public:
    /**
     * Tokenization result container.
     */
    struct TokenizedOutput {
        std::vector<int64_t> input_ids;       // Token IDs
        std::vector<int64_t> attention_mask;  // Attention mask (1=token, 0=pad)
        size_t actual_length;                 // Number of actual tokens (non-padding)
    };

    /**
     * Constructor.
     */
    BaiTokenizer();

    /**
     * Destructor.
     */
    ~BaiTokenizer() = default;

    // Disable copy
    BaiTokenizer(const BaiTokenizer&) = delete;
    BaiTokenizer& operator=(const BaiTokenizer&) = delete;

    // Allow move
    BaiTokenizer(BaiTokenizer&& other) noexcept = default;
    BaiTokenizer& operator=(BaiTokenizer&& other) noexcept = default;

    /**
     * Load tokenizer vocabulary from JSON file.
     * 
     * Expected format:
     * {
     *   "vocab": {
     *     "[PAD]": 0,
     *     "[CLS]": 101,
     *     "[SEP]": 102,
     *     ...
     *   },
     *   "special_tokens": { "[PAD]": 0, "[CLS]": 101, "[SEP]": 102 }
     * }
     * 
     * @param vocab_path Path to JSON vocabulary file
     * @return std::expected with void on success, error message on failure
     */
    std::expected<void, std::string> initialize(const std::string& vocab_path);

    /**
     * Encode text to token IDs and attention mask.
     * 
     * Performs:
     * 1. Text normalization and whitespace tokenization
     * 2. Subword tokenization using loaded vocabulary
     * 3. Special token insertion ([CLS] at start, [SEP] at end)
     * 4. Padding/truncation to max_seq_len
     * 5. Attention mask generation
     * 
     * @param text Input text string
     * @param max_seq_len Maximum sequence length (default 256)
     * @return std::expected containing TokenizedOutput on success, error on failure
     */
    std::expected<TokenizedOutput, std::string> encode(
        const std::string& text,
        size_t max_seq_len = 256
    );

    /**
     * Check if tokenizer is initialized.
     * @return true if vocabulary is loaded
     */
    bool is_initialized() const { return !vocab_.empty(); }

    /**
     * Get vocabulary size.
     * @return Number of tokens in vocabulary
     */
    size_t vocab_size() const { return vocab_.size(); }

    /**
     * Get special token IDs.
     */
    int64_t pad_token_id() const { return pad_token_id_; }
    int64_t cls_token_id() const { return cls_token_id_; }
    int64_t sep_token_id() const { return sep_token_id_; }
    int64_t unk_token_id() const { return unk_token_id_; }

private:
    // Token vocabulary: string -> token ID
    std::unordered_map<std::string, int64_t> vocab_;
    
    // Reverse vocabulary: token ID -> string (for debugging)
    std::unordered_map<int64_t, std::string> reverse_vocab_;

    // Special token IDs
    int64_t pad_token_id_ = 0;      // [PAD]
    int64_t cls_token_id_ = 101;    // [CLS]
    int64_t sep_token_id_ = 102;    // [SEP]
    int64_t unk_token_id_ = 100;    // [UNK]

    /**
     * Normalize text: lowercase, remove special chars, handle unicode.
     */
    std::string normalize_text(const std::string& text) const;

    /**
     * Split text on whitespace.
     */
    std::vector<std::string> whitespace_tokenize(const std::string& text) const;

    /**
     * Tokenize a single word using subword vocabulary.
     * Falls back to [UNK] if word not in vocabulary.
     */
    std::vector<int64_t> tokenize_word(const std::string& word) const;

    static bool decode_utf8(const std::string& text, size_t& offset, uint32_t& code_point);

    /**
     * Load and parse JSON vocabulary file.
     */
    std::expected<void, std::string> load_vocab_json(const std::string& vocab_path);
};

}  // namespace bai
