import express, { Request, Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = parseInt(process.env.PORT || '3000', 10);
const isProd = process.env.NODE_ENV === 'production';

app.use(express.json());

// Initialize Google GenAI SDK (Server-Side only)
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY || '',
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

// Endpoint: Health check and status
app.get('/api/health', (_req: Request, res: Response) => {
  res.json({
    status: 'ok',
    hasGeminiKey: Boolean(process.env.GEMINI_API_KEY),
    model: 'gemini-3.8-flash',
  });
});

// Endpoint: AI Biomolecule Docking & Affinity Analysis
app.post('/api/gemini/analyze-docking', async (req: Request, res: Response) => {
  try {
    const { target, ligand, simulationResult } = req.body;

    if (!target || !ligand || !simulationResult) {
      res.status(400).json({ error: 'Missing target, ligand, or simulation parameters.' });
      return;
    }

    if (!process.env.GEMINI_API_KEY) {
      // Intelligent bioinformatics fallback when API key is not yet set
      const fallbackAnalysis = generateBioinformaticsFallback(target, ligand, simulationResult);
      res.json({ analysis: fallbackAnalysis, isFallback: true });
      return;
    }

    const prompt = `
You are the lead computational chemist and quantum drug discovery specialist at QuantumDock Studio.
Analyze the following quantum-classical molecular docking screening results:

[TARGET PROTEIN]
Name: ${target.name} (${target.shortName})
PDB ID: ${target.pdbId}
Organism: ${target.organism}
Disease Area: ${target.diseaseArea}
Active Pocket: ${target.bindingSiteDescription}
Key Residues: ${target.keyResidues?.map((r: any) => `${r.name}${r.number} (${r.role})`).join(', ')}

[LIGAND CANDIDATE]
Name: ${ligand.name} ${ligand.tradeName ? `(${ligand.tradeName})` : ''}
SMILES: ${ligand.smiles}
Formula: ${ligand.formula}
MW: ${ligand.molecularWeight} g/mol
LogP: ${ligand.logP}
H-Bond Donors: ${ligand.hBondDonors} | Acceptors: ${ligand.hBondAcceptors}
Rotatable Bonds: ${ligand.rotatableBonds}
TPSA: ${ligand.tpsa} Å²
Mechanism: ${ligand.mechanism}

[SIMULATION METRICS]
Quantum VQE Ground State Binding Energy: ${simulationResult.vqeGroundStateEnergy} kcal/mol (${simulationResult.vqeEnergyHartree} Hartree)
Classical MM-GBSA Energy: ${simulationResult.classicalBindingEnergy} kcal/mol
Electronic Correlation Energy (ΔE_corr): ${simulationResult.electronicCorrelationEnergy} kcal/mol
Qubits: ${simulationResult.activeQubitsUsed} | Ansatz Depth: ${simulationResult.config.ansatzDepth} | Optimizer: ${simulationResult.config.optimizer}
Noise Mitigation: ${simulationResult.config.noiseMitigation}
Quantum State Fidelity: ${(simulationResult.quantumStateFidelity * 100).toFixed(2)}%
Predicted Dissociation Constant (Kd): ${simulationResult.predictedKdQuantum} ${simulationResult.kdUnit}
Affinity Tier: ${simulationResult.bindingAffinityClassification}
RMSD from Crystal Pose: ${simulationResult.rmsdFromCrystal} Å
Key Active Contacts: ${simulationResult.activeHydrogenBonds?.map((b: any) => `${b.donorResidue} ↔ ${b.acceptorAtom} (${b.type}, ${b.distanceAngstroms} Å, ${b.energyKcalMol} kcal/mol)`).join('; ')}

Perform an exhaustive, rigorous scientific assessment in valid JSON conforming to the requested schema.
Be specific about:
1. Why the VQE quantum correlation energy differs from the classical force field (e.g. electron pairing in covalent warheads, dispersion, or charge-transfer in the pocket).
2. ADMET & Drug-likeness profile according to Lipinski's Rule of 5, Veber rules, and blood-brain barrier permeability.
3. Specific chemical lead optimizations: give at least 2 distinct medicinal chemistry modifications to improve potency, selectivity, or metabolic stability.
4. Recommended biophysical experimental validation assays (e.g. SPR, ITC, Cryo-EM, mass spec).
`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            summary: {
              type: Type.STRING,
              description: 'Executive scientific summary of binding potential and quantum docking score',
            },
            quantumInsight: {
              type: Type.OBJECT,
              properties: {
                correlationImpact: { type: Type.STRING },
                electronicOverlaps: { type: Type.STRING },
                accuracyAdvantage: { type: Type.STRING },
              },
              required: ['correlationImpact', 'electronicOverlaps', 'accuracyAdvantage'],
            },
            drugLikeness: {
              type: Type.OBJECT,
              properties: {
                lipinskiCompliance: { type: Type.BOOLEAN },
                lipinskiViolations: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
                bioavailabilityScore: { type: Type.NUMBER },
                bloodBrainBarrier: {
                  type: Type.STRING,
                  enum: ['High', 'Moderate', 'Low'],
                },
                admetAssessment: { type: Type.STRING },
              },
              required: ['lipinskiCompliance', 'lipinskiViolations', 'bioavailabilityScore', 'bloodBrainBarrier', 'admetAssessment'],
            },
            toxicityAndRisks: {
              type: Type.OBJECT,
              properties: {
                hergCardiotoxicity: {
                  type: Type.STRING,
                  enum: ['Low', 'Moderate', 'High'],
                },
                hepatotoxicityRisk: {
                  type: Type.STRING,
                  enum: ['Low', 'Moderate', 'High'],
                },
                cypInteractions: { type: Type.STRING },
              },
              required: ['hergCardiotoxicity', 'hepatotoxicityRisk', 'cypInteractions'],
            },
            leadOptimizations: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  targetSite: { type: Type.STRING },
                  modification: { type: Type.STRING },
                  rationale: { type: Type.STRING },
                  predictedAffinityGain: { type: Type.STRING },
                },
                required: ['targetSite', 'modification', 'rationale', 'predictedAffinityGain'],
              },
            },
            experimentalValidation: {
              type: Type.OBJECT,
              properties: {
                recommendedAssays: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
                crystallographyConditions: { type: Type.STRING },
              },
              required: ['recommendedAssays', 'crystallographyConditions'],
            },
          },
          required: [
            'summary',
            'quantumInsight',
            'drugLikeness',
            'toxicityAndRisks',
            'leadOptimizations',
            'experimentalValidation',
          ],
        },
      },
    });

    const parsed = JSON.parse(response.text || '{}');
    res.json({ analysis: parsed, isFallback: false });
  } catch (error: any) {
    console.error('Gemini analyze-docking error:', error);
    // Fall back smoothly to ensure app never hangs
    const fallbackAnalysis = generateBioinformaticsFallback(
      req.body.target,
      req.body.ligand,
      req.body.simulationResult
    );
    res.json({ analysis: fallbackAnalysis, isFallback: true, error: error.message });
  }
});

// Endpoint: Interactive Biomolecule Q&A Chat
app.post('/api/gemini/chat-docking', async (req: Request, res: Response) => {
  try {
    const { message, context, chatHistory } = req.body;

    if (!message) {
      res.status(400).json({ error: 'Message is required' });
      return;
    }

    if (!process.env.GEMINI_API_KEY) {
      res.json({
        reply: `[QuantumDock AI]: Under current simulation parameters for ${context?.target?.shortName || 'the target'} with ${context?.ligand?.name || 'the ligand'}, the calculated VQE ground state energy of ${context?.simulationResult?.vqeGroundStateEnergy || '-9.2'} kcal/mol indicates favorable binding kinetics. Note: To enable live generative reasoning with full Gemini 3.8 models, provide your GEMINI_API_KEY in the Secrets panel.`,
      });
      return;
    }

    const systemPrompt = `You are QuantumDock Assistant, an expert AI biophysicist and medicinal chemist.
Context:
Target: ${context?.target?.name || 'Unknown'} (PDB: ${context?.target?.pdbId || 'N/A'})
Ligand: ${context?.target?.name || 'Candidate'} (MW: ${context?.ligand?.molecularWeight || 'N/A'})
VQE Ground Energy: ${context?.simulationResult?.vqeGroundStateEnergy || 'N/A'} kcal/mol
Classical MM Energy: ${context?.simulationResult?.classicalBindingEnergy || 'N/A'} kcal/mol
Electronic Correlation (ΔE_corr): ${context?.simulationResult?.electronicCorrelationEnergy || 'N/A'} kcal/mol
Predicted Kd: ${context?.simulationResult?.predictedKdQuantum || 'N/A'} ${context?.simulationResult?.kdUnit || 'nM'}

Respond directly, scientifically, and concisely to the researcher's question. Focus on quantum chemical insights, orbital interactions, synthetic feasibility, and medicinal chemistry optimization.`;

    const contents = [
      { role: 'user', parts: [{ text: `${systemPrompt}\n\nUser Question: ${message}` }] },
    ];

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents,
    });

    res.json({ reply: response.text });
  } catch (error: any) {
    console.error('Gemini chat error:', error);
    res.status(500).json({
      error: 'Failed to generate response',
      details: error.message,
    });
  }
});

// Helper for deterministic bioinformatics fallback
function generateBioinformaticsFallback(target: any, ligand: any, sim: any) {
  const energy = sim.vqeGroundStateEnergy;
  const correlation = sim.electronicCorrelationEnergy;
  const isGoodAffinity = energy < -7.5;

  return {
    summary: `${ligand.name} exhibits ${isGoodAffinity ? 'high-potency favorable' : 'moderate'} binding affinity towards ${target.shortName} with a simulated quantum ground-state energy of ${energy} kcal/mol (Kd ≈ ${sim.predictedKdQuantum} ${sim.kdUnit}).`,
    quantumInsight: {
      correlationImpact: `The VQE algorithm resolves an electronic correlation contribution of ${correlation} kcal/mol beyond Hartree-Fock reference, capturing non-covalent dispersion and electron pairing across the catalytic pocket.`,
      electronicOverlaps: `Multi-orbital mapping with ${sim.activeQubitsUsed} qubits accurately accounts for polarized hydrogen bonding and partial charge transfer with ${target.keyResidues?.[0]?.name || 'catalytic residues'}.`,
      accuracyAdvantage: `Classical force fields over/under-estimate binding due to fixed point charges, whereas VQE achieves within ±${sim.chemicalAccuracyThreshold} kcal/mol chemical accuracy.`,
    },
    drugLikeness: {
      lipinskiCompliance: ligand.molecularWeight <= 500 && ligand.logP <= 5.0 && ligand.hBondDonors <= 5 && ligand.hBondAcceptors <= 10,
      lipinskiViolations: ligand.molecularWeight > 500 ? ['MW > 500 Da'] : [],
      bioavailabilityScore: ligand.molecularWeight < 520 && ligand.rotatableBonds <= 10 ? 85 : 65,
      bloodBrainBarrier: ligand.tpsa < 90 && ligand.logP > 1.5 && ligand.logP < 3.5 ? 'Moderate' : 'Low',
      admetAssessment: `With a TPSA of ${ligand.tpsa} Å² and LogP of ${ligand.logP}, the compound exhibits favorable oral absorption characteristics and balanced lipophilicity.`,
    },
    toxicityAndRisks: {
      hergCardiotoxicity: ligand.logP > 4.0 ? 'Moderate' : 'Low',
      hepatotoxicityRisk: ligand.molecularWeight > 550 ? 'Moderate' : 'Low',
      cypInteractions: 'Likely substrate for CYP3A4 oxidation; mild reversible inhibition anticipated.',
    },
    leadOptimizations: [
      {
        targetSite: 'Hydrophobic subpocket pocket interface',
        modification: 'Introduce ortho-fluorine or cyclopropyl substituent',
        rationale: 'Fills the hydrophobic void and increases local van der Waals packing density without increasing polar desolvation penalty.',
        predictedAffinityGain: '-0.8 to -1.4 kcal/mol (3x to 5x affinity increase)',
      },
      {
        targetSite: 'Hydrogen bond donor terminus',
        modification: 'Replace secondary amide with bioisosteric heterocyclic surrogate (e.g. 1,2,4-oxadiazole)',
        rationale: 'Improves metabolic stability against amidases while preserving directional electrostatic bonding.',
        predictedAffinityGain: '-0.5 kcal/mol with 2.2x extended metabolic half-life',
      },
    ],
    experimentalValidation: {
      recommendedAssays: [
        'Surface Plasmon Resonance (SPR) binding kinetics (ka, kd, KD)',
        'Isothermal Titration Calorimetry (ITC) to dissect enthalpy vs entropy',
        'High-resolution X-ray co-crystallography (target resolution < 2.0 Å)',
      ],
      crystallographyConditions: 'Tris-HCl pH 7.5, 20% PEG 3350, 200 mM sodium chloride at 20 °C in sitting-drop vapor diffusion.',
    },
  };
}

// Dev & Production serving
if (!isProd) {
  const { createServer: createViteServer } = await import('vite');
  const vite = await createViteServer({
    server: { middlewareMode: true },
    appType: 'spa',
  });
  app.use(vite.middlewares);
} else {
  app.use(express.static(path.resolve(__dirname, 'dist')));
  app.get('*', (_req: Request, res: Response) => {
    res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
  });
}

app.listen(port, '0.0.0.0', () => {
  console.log(`QuantumDock Server running at http://0.0.0.0:${port}`);
});
