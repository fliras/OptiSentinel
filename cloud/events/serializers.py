from rest_framework import serializers


ALLOWED_MIME = {"image/jpeg", "image/png"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  


class EventRequestSerializer(serializers.Serializer):

    image = serializers.ImageField()
    camera_id = serializers.CharField(max_length=64)
    setor = serializers.CharField(max_length=128)
    timestamp = serializers.DateTimeField()
    deteccao = serializers.CharField(max_length=128)

    def validate_image(self, value):
        if value.size > MAX_IMAGE_BYTES:
            raise serializers.ValidationError(
                f"Imagem excede {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
            )
        if value.content_type not in ALLOWED_MIME:
            raise serializers.ValidationError(
                f"Formato não suportado: {value.content_type}. Use JPEG ou PNG."
            )
        return value

    def to_metadata(self) -> dict:
        data = self.validated_data
        return {
            "camera_id": data["camera_id"],
            "setor": data["setor"],
            "timestamp": data["timestamp"].isoformat(),
            "deteccao": data["deteccao"],
        }


class EventResponseSerializer(serializers.Serializer):

    resumo = serializers.CharField()
    descricao_detalhada = serializers.CharField()
    severidade = serializers.ChoiceField(choices=["baixa", "media", "alta"])
    pessoas_envolvidas = serializers.IntegerField()
    riscos_identificados = serializers.ListField(child=serializers.CharField())
    confianca = serializers.FloatField(min_value=0, max_value=1)