"""
test_ollama.py — Test rapide de l'intégration Ollama dans LLMGatewayTool
Lancez ce script pour vérifier que le backend Ollama est bien détecté.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from scripts.agents.core.tools import LLMGatewayTool

print("=" * 60)
print("  TEST INTÉGRATION OLLAMA — LLMGatewayTool")
print("=" * 60)

tool = LLMGatewayTool()

print(f"\n  Backend actif : {tool.active_backend}")
print(f"  Modèle        : {tool.model_name}")
print(f"  En ligne      : {tool.is_online}")

if tool.is_online:
    print("\n  Envoi d'un prompt test...")
    result = tool.execute(prompt="Dis bonjour en 1 phrase en français.")
    print(f"\n  Réponse  : {result['text']}")
    print(f"  Latence  : {result['latency_ms']} ms")
    print(f"  Backend  : {result['backend']}")
else:
    print("\n  ⚠️  Aucun backend disponible. Vérifiez qu'Ollama est lancé.")

print("\n" + "=" * 60)
