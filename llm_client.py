"""
LLM客户端 - 连接自定义API的大语言模型
"""
import json
import requests
from typing import List, Dict, Generator
from dataclasses import dataclass
import logging


@dataclass
class Message:
    """消息类"""
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> Dict:
        return {"role": self.role, "content": self.content}


class LLMClient:
    """自定义API的LLM客户端"""
    
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        timeout: int = 60
    ):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        print(f"LLM客户端初始化完成，API端点: {self.api_base}, 模型: {self.model_name}")
        
    def generate_response(
        self,
        messages: List[Message],
        stream: bool = False
    ) -> Generator[str, None, None] | str:
        """生成响应"""
        payload = {
            "model": self.model_name,
            "messages": [msg.to_dict() for msg in messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 强制关闭流式
        payload["stream"] = False
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout
            )
            response.raise_for_status()
    
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")

        except requests.exceptions.RequestException as e:
            print(f"LLM 请求失败: {e}")
            if stream:
                return iter([])
            return ""