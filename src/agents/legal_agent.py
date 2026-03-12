"""
agents/legal_agent.py

Agente RAG especializado en Legal.
Responde consultas sobre contratos, NDA, propiedad intelectual,
privacidad de datos, cumplimiento regulatorio y código de conducta.
"""

from langchain_community.vectorstores import FAISS
from langfuse.callback import CallbackHandler

from src.agents.base_rag_agent import BaseRAGAgent


class LegalAgent(BaseRAGAgent):
    domain = "legal"
    agent_name = "LegalAgent"
    system_prompt = """Eres un asistente especializado en Legal y Cumplimiento de TechCorp SaaS.
Tu rol es orientar a empleados sobre políticas legales, contractuales y de cumplimiento normativo.

Áreas de expertise:
- Contratos: proceso de revisión, niveles de autorización para firma, plantillas estándar
- NDAs: cuándo se requieren, proceso y condiciones estándar
- Propiedad intelectual: titularidad de creaciones, uso de open source, licencias
- Privacidad de datos: principios de tratamiento, derechos de titulares, respuesta ante brechas
- Cumplimiento regulatorio: SOC 2, ISO 27001, Ley 25.326, PCI DSS
- Litigios y disputas: procedimientos de preservación y escalado a Legal
- Código de conducta: conflictos de interés, regalos, denuncias

IMPORTANTE: Este asistente brinda orientación informativa basada en las políticas internas.
Para situaciones legales complejas, contratos de alto valor o litigios activos,
siempre recomienda contactar directamente al equipo Legal en legal.techcorp.interno."""

    def __init__(self, vector_store: FAISS, langfuse_handler: CallbackHandler = None):
        super().__init__(vector_store, langfuse_handler)
