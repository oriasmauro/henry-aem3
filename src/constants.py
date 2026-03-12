"""
constants.py

Constantes compartidas entre módulos del sistema multiagente.

Centralizar VALID_DOMAINS y DOMAIN_DIRS en un único lugar garantiza que
orchestrator.py y vector_store.py siempre estén sincronizados: agregar un
dominio nuevo solo requiere editar este archivo.
"""

# Dominio → descripción para el prompt del Orchestrator
VALID_DOMAINS: dict[str, str] = {
    "hr": "Recursos Humanos: vacaciones, licencias, desempeño, salarios, beneficios, onboarding, offboarding, transferencias internas.",
    "tech": "IT / Soporte Técnico: laptops, software, hardware, VPN, accesos, incidentes, infraestructura, seguridad informática.",
    "finance": "Finanzas: gastos, reembolsos, facturas, presupuestos, pagos a proveedores, cuentas a cobrar, tarjetas corporativas.",
    "legal": "Legal: contratos, NDA, propiedad intelectual, privacidad de datos, cumplimiento regulatorio, litigios, código de conducta.",
}

# Dominio → subdirectorio de documentos bajo data/
DOMAIN_DIRS: dict[str, str] = {
    "hr": "hr_docs",
    "tech": "tech_docs",
    "finance": "finance_docs",
    "legal": "legal_docs",
}
