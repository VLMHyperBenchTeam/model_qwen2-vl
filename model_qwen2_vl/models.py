from typing import Any, List

import torch
from model_interface.model_interface import ModelInterface
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class Qwen2VLModel(ModelInterface):
    """
    Модель для обработки изображений и генерации текстовых ответов на основе архитектуры Qwen2-VL.
    Поддерживает работу с изображениями и видео, включая мульти-модальные запросы.

    Args:
        model_name (str, optional): Название предобученной модели из семейства Qwen2-VL.
            По умолчанию "Qwen2-VL-2B-Instruct".
        system_prompt (str, optional): Системный промпт для настройки поведения модели.
            По умолчанию пустая строка.
        cache_dir (str, optional): Директория для кэширования моделей. По умолчанию "model_cache".

    Attributes:
        model (Qwen2VLForConditionalGeneration): Загруженная модель HuggingFace.
        processor (AutoProcessor): Процессор для предобработки данных.
        min_pixels (int): Минимальное разрешение изображения в пикселях.
        max_pixels (int): Максимальное разрешение изображения в пикселях.
    """

    def __init__(
        self,
        model_name="Qwen2-VL-2B-Instruct",
        system_prompt="",
        cache_dir="model_cache",
    ):
        super().__init__(model_name, system_prompt, cache_dir)
        self.framework = "Hugging_Face"  # явное указание фреймворка

        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1536 * 28 * 28  # 1280 * 28 * 28

        # default: Load the model on the available device(s)
        model_path = f"Qwen/{model_name}"
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                cache_dir=self.cache_dir,
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")

        self.processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)

    def get_message(self, image, question):
        """
        Формирует структуру сообщения для взаимодействия с моделью, включая изображение и текстовый запрос.

        Args:
            image (Any): Изображение в поддерживаемом формате (например, PIL.Image или np.ndarray).
                Будет автоматически масштабировано в диапазоне min_pixels-max_pixels [[7]].
            question (str): Текстовый вопрос или инструкция для обработки изображения.

        Returns:
            dict: Сообщение в формате:
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": self.min_pixels,
                            "max_pixels": self.max_pixels,
                        },
                        {"type": "text", "text": question},
                    ],
                }

        Notes:
            * Использует динамическое масштабирование изображений для оптимизации потребления памяти [[7]]
            * Поддерживает все форматы изображений, совместимые с `qwen_vl_utils.process_vision_info`
        """
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                },
                {"type": "text", "text": question},
            ],
        }

        return message

    def predict_on_image(self, image: Any, question: str) -> str:
        """
        Генерирует текстовый ответ на основе одного изображения и вопроса.
        
        Args:
            image: Изображение в поддерживаемом формате (PIL.Image, np.ndarray, bytes)
            question: Текстовый вопрос к содержимому изображения
            
        Returns:
            str: Сгенерированный текстовый ответ
            
        Raises:
            ValueError: Некорректный формат изображения
            RuntimeError: Ошибка инференса или превышение GPU-памяти
        """
        try:
            # Формирование сообщения
            messages = [self.get_message(image, question)]

            # Предобработка данных
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Обработка визуальной информации
            image_inputs, video_inputs = process_vision_info(messages)

            # Подготовка входных тензоров
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            # Генерация с обработкой ошибок памяти
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=512, do_sample=False
                )

            # Постобработка результата
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return output_text

        except ValueError as ve:
            raise ValueError(f"Некорректные входные данные: {str(ve)}") from ve
        except torch.cuda.OutOfMemoryError as oom:
            raise RuntimeError("Превышен лимит GPU-памяти") from oom
        except Exception as e:
            raise RuntimeError(f"Ошибка инференса: {str(e)}") from e

def predict_on_images(self, images: List[Any], question: str) -> str:
    """
    Генерирует текстовый ответ на основе нескольких изображений и заданного вопроса.
    
    Args:
        images: Список изображений в поддерживаемых форматах (PIL.Image, np.ndarray, bytes)
        question: Текстовый вопрос, связанный с содержимым изображений
        
    Returns:
        str: Сгенерированный текстовый ответ
        
    Raises:
        ValueError: Некорректный формат данных или пустой список изображений
        RuntimeError: Ошибка инференса или превышение GPU-памяти
    """
    try:
        # Проверка входных данных
        if not images:
            raise ValueError("Список изображений не может быть пустым")
            
        # Формирование сообщения с несколькими изображениями
        messages = [{
            "role": "user",
            "content": [
                *[
                    {
                        "type": "image",
                        "image": img,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    } 
                    for img in images
                ],
                {"type": "text", "text": question}
            ]
        }]
        
        # Предобработка данных
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Обработка входных данных
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Подготовка входных тензоров
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Генерация с обработкой ошибок памяти
        with torch.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
            
        # Постобработка результата
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
        
    except ValueError as ve:
        raise ValueError(f"Ошибка в данных: {str(ve)}") from ve
    except torch.cuda.OutOfMemoryError as oom:
        raise RuntimeError("Превышен лимит GPU-памяти") from oom
    except Exception as e:
        raise RuntimeError(f"Ошибка инференса: {str(e)}") from e
