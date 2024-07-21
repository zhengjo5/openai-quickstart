from book import ContentType

class Model:
    def make_text_prompt(self, text: str, origin_language: str, target_language: str) -> str:
        return f"从{origin_language}翻译为{target_language}，保持间距（空格，分隔符，换行）：{text}"

    def make_table_prompt(self, table: str, origin_language: str, target_language: str) -> str:
        return f"从{origin_language}翻译为{target_language}，保持间距（空格，分隔符，换行），以表格的格式返回：\n{table}"

    def translate_prompt(self, content, origin_language: str, target_language: str) -> str:
        if content.content_type == ContentType.TEXT:
            return self.make_text_prompt(content.original, origin_language, target_language)
        elif content.content_type == ContentType.TABLE:
            return self.make_table_prompt(content.get_original_as_str(), origin_language, target_language)

    def make_request(self, prompt):
        raise NotImplementedError("子类必须实现 make_request 方法")
