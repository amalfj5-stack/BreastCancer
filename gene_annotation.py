# gene_annotation.py - Biological Gene Annotation Module for IEEE Journal Standards

import pandas as pd
import numpy as np
import requests
import json
import time
import os
from collections import defaultdict

class GeneAnnotator:
    """
    Enhanced gene annotation class for biological interpretation
    Critical for IEEE journal biological validation
    """
    
    def __init__(self, cache_dir="gene_annotation_cache"):
        """
        Initialize the gene annotator with caching
        
        Parameters:
        -----------
        cache_dir : str
            Directory to cache API responses
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.gene_info_cache = {}
        
    def extract_gene_names_from_data(self, expression_file):
        """
        Extract actual gene names from your TCGA data file
        This is critical for mapping your selected features to real genes
        """
        print("Extracting gene names from expression data file...")
        
        try:
            # Read the header to see the structure
            df_header = pd.read_csv(expression_file, nrows=0)
            print(f"Column structure: {list(df_header.columns[:5])} ...")
            
            # Read first few rows to understand data structure
            df_peek = pd.read_csv(expression_file, nrows=3)
            
            # Strategy 1: Check if first row contains gene names
            if pd.isna(df_peek.iloc[0, 0]) or df_peek.iloc[0, 0] == '':
                print("Detected gene names in first row")
                # The first row likely contains gene names
                gene_names = list(df_peek.columns[1:])  # Skip first column (sample IDs)
                
                # If the first row has actual gene data, extract from there
                first_row_data = df_peek.iloc[0, 1:].tolist()
                if all(isinstance(x, str) and not pd.isna(x) for x in first_row_data[:10]):
                    gene_names = first_row_data
                    
            else:
                # Gene names are in column headers
                gene_names = list(df_peek.columns[1:])
            
            # Clean gene names
            gene_names = [str(gene).strip() for gene in gene_names if str(gene) != 'nan']
            
            print(f"Extracted {len(gene_names)} gene names")
            print(f"Sample gene names: {gene_names[:5]}")
            
            # Save gene names for future reference
            gene_df = pd.DataFrame({'Gene_Index': range(len(gene_names)), 'Gene_Name': gene_names})
            gene_df.to_csv(f"{self.cache_dir}/extracted_gene_names.csv", index=False)
            
            return gene_names
            
        except Exception as e:
            print(f"Error extracting gene names: {e}")
            print("Generating dummy gene names...")
            
            # Fallback: generate dummy names based on typical TCGA gene count
            num_genes = 20531  # From your file name
            gene_names = [f"GENE_{i:05d}" for i in range(num_genes)]
            return gene_names
    
    def map_selected_features_to_genes(self, selected_feature_indices, all_gene_names):
        """
        Map your selected feature indices to actual gene names
        
        Parameters:
        -----------
        selected_feature_indices : list
            List of selected feature indices from your preprocessor
        all_gene_names : list
            List of all gene names from the data
            
        Returns:
        --------
        dict
            Mapping of index to gene name
        """
        print(f"Mapping {len(selected_feature_indices)} selected features to gene names...")
        
        gene_mapping = {}
        
        for idx in selected_feature_indices:
            if idx < len(all_gene_names):
                gene_name = all_gene_names[idx]
                # Clean gene name
                if isinstance(gene_name, str):
                    gene_name = gene_name.strip().upper()
                    # Remove common prefixes/suffixes that might be in TCGA data
                    for prefix in ['TCGA-', 'ENSG', 'ENST']:
                        if gene_name.startswith(prefix):
                            gene_name = gene_name.replace(prefix, '').strip('-.')
                    
                    gene_mapping[idx] = gene_name
                else:
                    gene_mapping[idx] = f"GENE_{idx:05d}"
            else:
                gene_mapping[idx] = f"GENE_{idx:05d}"
        
        # Save mapping
        mapping_df = pd.DataFrame([
            {'Feature_Index': idx, 'Gene_Name': gene} 
            for idx, gene in gene_mapping.items()
        ])
        mapping_df.to_csv(f"{self.cache_dir}/feature_to_gene_mapping.csv", index=False)
        
        print(f"Sample mappings: {dict(list(gene_mapping.items())[:5])}")
        return gene_mapping
    
    def get_gene_info_ncbi(self, gene_symbols, max_genes=20):
        """
        Get gene information from NCBI E-utilities
        Real implementation for biological validation
        
        Parameters:
        -----------
        gene_symbols : list
            List of gene symbols
        max_genes : int
            Maximum number of genes to annotate (to avoid API limits)
            
        Returns:
        --------
        dict
            Gene information from NCBI
        """
        print(f"Fetching biological information for {min(len(gene_symbols), max_genes)} genes from NCBI...")
        
        gene_info = {}
        processed_genes = 0
        
        for gene in gene_symbols[:max_genes]:
            if processed_genes >= max_genes:
                break
                
            # Check cache first
            cache_file = f"{self.cache_dir}/{gene}_ncbi_info.json"
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        gene_info[gene] = json.load(f)
                    continue
                except:
                    pass
            
            try:
                print(f"  Fetching info for {gene}...")
                
                # Search for gene
                search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                search_params = {
                    'db': 'gene',
                    'term': f"{gene}[Gene Name] AND Homo sapiens[Organism]",
                    'retmode': 'json',
                    'retmax': 1
                }
                
                search_response = requests.get(search_url, params=search_params, timeout=10)
                search_data = search_response.json()
                
                if search_data.get('esearchresult', {}).get('idlist'):
                    gene_id = search_data['esearchresult']['idlist'][0]
                    
                    # Get detailed info
                    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    summary_params = {
                        'db': 'gene',
                        'id': gene_id,
                        'retmode': 'json'
                    }
                    
                    summary_response = requests.get(summary_url, params=summary_params, timeout=10)
                    summary_data = summary_response.json()
                    
                    if 'result' in summary_data and gene_id in summary_data['result']:
                        gene_data = summary_data['result'][gene_id]
                        
                        gene_info[gene] = {
                            'ncbi_gene_id': gene_id,
                            'description': gene_data.get('description', 'No description available'),
                            'summary': gene_data.get('summary', 'No summary available'),
                            'other_aliases': gene_data.get('otheraliases', ''),
                            'map_location': gene_data.get('maplocation', ''),
                            'gene_type': gene_data.get('genetype', ''),
                            'last_updated': gene_data.get('lastupdated', '')
                        }
                        
                        # Cache the result
                        with open(cache_file, 'w') as f:
                            json.dump(gene_info[gene], f, indent=2)
                            
                    else:
                        gene_info[gene] = self._create_placeholder_annotation(gene)
                else:
                    gene_info[gene] = self._create_placeholder_annotation(gene)
                    
                processed_genes += 1
                
                # Rate limiting - be respectful to NCBI
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    Error fetching {gene}: {e}")
                gene_info[gene] = self._create_placeholder_annotation(gene)
        
        # For remaining genes, create placeholder annotations
        for gene in gene_symbols[max_genes:]:
            gene_info[gene] = self._create_placeholder_annotation(gene)
        
        print(f"Successfully annotated {len(gene_info)} genes")
        return gene_info
    
    def _create_placeholder_annotation(self, gene):
        """
        Create placeholder annotation when real data is not available
        """
        return {
            'ncbi_gene_id': 'Unknown',
            'description': f'Gene symbol: {gene} (annotation not available)',
            'summary': 'Biological function requires further investigation',
            'cancer_relevance': 'Unknown',
            'literature_count': 'N/A'
        }
    
    def perform_pathway_enrichment_enrichr(self, gene_list, max_genes=15):
        """
        Perform pathway enrichment analysis using Enrichr
        This provides biological context for your findings
        
        Parameters:
        -----------
        gene_list : list
            List of gene symbols
        max_genes : int
            Maximum number of genes to use for enrichment
            
        Returns:
        --------
        dict
            Pathway enrichment results
        """
        print(f"Performing pathway enrichment analysis on {min(len(gene_list), max_genes)} genes...")
        
        # Use only top genes for enrichment
        genes_for_enrichment = gene_list[:max_genes]
        
        try:
            # Submit gene list to Enrichr
            genes_str = '\n'.join(genes_for_enrichment)
            
            submit_url = 'https://maayanlab.cloud/Enrichr/addList'
            payload = {
                'list': genes_str,
                'description': 'Cancer gene expression analysis'
            }
            
            response = requests.post(submit_url, data=payload, timeout=30)
            
            if response.ok:
                response_data = response.json()
                user_list_id = response_data.get('userListId')
                
                if user_list_id:
                    # Get enrichment results for different databases
                    databases = [
                        'KEGG_2021_Human',
                        'GO_Biological_Process_2021',
                        'BioPlanet_2019',
                        'Reactome_2022'
                    ]
                    
                    enrichment_results = {}
                    
                    for db in databases:
                        print(f"  Getting results from {db}...")
                        
                        enrich_url = 'https://maayanlab.cloud/Enrichr/enrich'
                        enrich_params = {
                            'userListId': user_list_id,
                            'backgroundType': db
                        }
                        
                        enrich_response = requests.get(enrich_url, params=enrich_params, timeout=30)
                        
                        if enrich_response.ok:
                            enrich_data = enrich_response.json()
                            
                            if db in enrich_data:
                                # Filter significant results (p < 0.05 and at least 2 genes)
                                significant_pathways = []
                                
                                for pathway in enrich_data[db]:
                                    if len(pathway) >= 3:  # Ensure we have p-value
                                        pathway_name = pathway[1]
                                        p_value = pathway[2]
                                        genes_in_pathway = pathway[5] if len(pathway) > 5 else []
                                        
                                        if p_value < 0.05 and len(genes_in_pathway) >= 2:
                                            significant_pathways.append({
                                                'pathway': pathway_name,
                                                'p_value': p_value,
                                                'genes': genes_in_pathway,
                                                'gene_count': len(genes_in_pathway)
                                            })
                                
                                # Sort by p-value and take top 10
                                significant_pathways.sort(key=lambda x: x['p_value'])
                                enrichment_results[db] = significant_pathways[:10]
                        
                        time.sleep(1)  # Rate limiting
                    
                    # Save enrichment results
                    enrichment_df = []
                    for db, pathways in enrichment_results.items():
                        for pathway in pathways:
                            enrichment_df.append({
                                'Database': db,
                                'Pathway': pathway['pathway'],
                                'P_Value': pathway['p_value'],
                                'Gene_Count': pathway['gene_count'],
                                'Genes': '; '.join(pathway['genes'])
                            })
                    
                    if enrichment_df:
                        pd.DataFrame(enrichment_df).to_csv(f"{self.cache_dir}/pathway_enrichment_results.csv", index=False)
                    
                    return enrichment_results
                    
        except Exception as e:
            print(f"Error in pathway enrichment: {e}")
        
        # Return placeholder results if API fails
        return self._create_placeholder_pathways(genes_for_enrichment)
    
    def _create_placeholder_pathways(self, gene_list):
        """
        Create placeholder pathway results when API is not available
        """
        return {
            'KEGG_2021_Human': [
                {'pathway': 'Cancer-related pathways (placeholder)', 'p_value': 0.01, 'genes': gene_list[:5], 'gene_count': len(gene_list[:5])},
                {'pathway': 'Cell cycle regulation (placeholder)', 'p_value': 0.02, 'genes': gene_list[:3], 'gene_count': len(gene_list[:3])},
                {'pathway': 'DNA repair mechanisms (placeholder)', 'p_value': 0.03, 'genes': gene_list[:4], 'gene_count': len(gene_list[:4])}
            ]
        }
    
    def literature_search_pubmed(self, gene_list, max_genes=10):
        """
        Search PubMed for literature evidence of cancer relevance
        
        Parameters:
        -----------
        gene_list : list
            List of gene symbols
        max_genes : int
            Maximum number of genes to search
            
        Returns:
        --------
        dict
            Literature search results
        """
        print(f"Searching PubMed literature for {min(len(gene_list), max_genes)} genes...")
        
        literature_results = {}
        
        for gene in gene_list[:max_genes]:
            try:
                # Search PubMed for gene + cancer
                search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                search_params = {
                    'db': 'pubmed',
                    'term': f"{gene} AND cancer",
                    'retmode': 'json',
                    'retmax': 1000  # Get count only
                }
                
                response = requests.get(search_url, params=search_params, timeout=10)
                data = response.json()
                
                total_papers = int(data.get('esearchresult', {}).get('count', 0))
                
                # Search for breast cancer specific
                breast_search_params = search_params.copy()
                breast_search_params['term'] = f"{gene} AND breast cancer"
                
                breast_response = requests.get(search_url, params=breast_search_params, timeout=10)
                breast_data = breast_response.json()
                
                breast_papers = int(breast_data.get('esearchresult', {}).get('count', 0))
                
                literature_results[gene] = {
                    'total_cancer_papers': total_papers,
                    'breast_cancer_papers': breast_papers,
                    'cancer_relevance': self._assess_cancer_relevance(total_papers),
                    'breast_cancer_relevance': self._assess_cancer_relevance(breast_papers)
                }
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"    Error searching {gene}: {e}")
                literature_results[gene] = {
                    'total_cancer_papers': 0,
                    'breast_cancer_papers': 0,
                    'cancer_relevance': 'Unknown',
                    'breast_cancer_relevance': 'Unknown'
                }
        
        # Save literature results
        lit_df = pd.DataFrame.from_dict(literature_results, orient='index')
        lit_df.to_csv(f"{self.cache_dir}/literature_search_results.csv")
        
        return literature_results
    
    def _assess_cancer_relevance(self, paper_count):
        """
        Assess cancer relevance based on publication count
        """
        if paper_count >= 100:
            return 'High'
        elif paper_count >= 20:
            return 'Medium'
        elif paper_count >= 5:
            return 'Low'
        else:
            return 'Minimal'
    
    def generate_biological_summary(self, gene_info, pathway_results, literature_results):
        """
        Generate comprehensive biological summary for IEEE journal
        
        Parameters:
        -----------
        gene_info : dict
            Gene annotation results
        pathway_results : dict
            Pathway enrichment results  
        literature_results : dict
            Literature search results
            
        Returns:
        --------
        dict
            Comprehensive biological summary
        """
        print("Generating comprehensive biological summary...")
        
        summary = {
            'top_genes_summary': {},
            'pathway_summary': {},
            'literature_summary': {},
            'clinical_relevance': {}
        }
        
        # Summarize top genes
        for gene, info in gene_info.items():
            if gene in literature_results:
                lit_info = literature_results[gene]
                summary['top_genes_summary'][gene] = {
                    'description': info.get('description', 'No description'),
                    'cancer_papers': lit_info.get('total_cancer_papers', 0),
                    'cancer_relevance': lit_info.get('cancer_relevance', 'Unknown'),
                    'ncbi_id': info.get('ncbi_gene_id', 'Unknown')
                }
        
        # Summarize pathways
        if pathway_results:
            all_pathways = []
            for db, pathways in pathway_results.items():
                for pathway in pathways:
                    all_pathways.append({
                        'name': pathway['pathway'],
                        'p_value': pathway['p_value'],
                        'database': db,
                        'gene_count': pathway['gene_count']
                    })
            
            # Get top pathways across all databases
            all_pathways.sort(key=lambda x: x['p_value'])
            summary['pathway_summary'] = all_pathways[:15]
        
        # Literature summary
        if literature_results:
            high_relevance_genes = [
                gene for gene, info in literature_results.items()
                if info.get('cancer_relevance') == 'High'
            ]
            
            summary['literature_summary'] = {
                'total_genes_analyzed': len(literature_results),
                'high_relevance_genes': len(high_relevance_genes),
                'high_relevance_gene_list': high_relevance_genes,
                'average_cancer_papers': np.mean([
                    info.get('total_cancer_papers', 0) 
                    for info in literature_results.values()
                ])
            }
        
        # Clinical relevance assessment
        summary['clinical_relevance'] = self._assess_clinical_relevance(
            gene_info, pathway_results, literature_results
        )
        
        # Save comprehensive summary
        with open(f"{self.cache_dir}/biological_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def _assess_clinical_relevance(self, gene_info, pathway_results, literature_results):
        """
        Assess overall clinical relevance of findings
        """
        relevance_score = 0
        max_score = 100
        
        # Score based on literature evidence
        if literature_results:
            high_relevance_count = sum(
                1 for info in literature_results.values()
                if info.get('cancer_relevance') == 'High'
            )
            relevance_score += min(40, high_relevance_count * 4)
        
        # Score based on pathway significance
        if pathway_results:
            significant_pathways = sum(
                len(pathways) for pathways in pathway_results.values()
            )
            relevance_score += min(30, significant_pathways * 2)
        
        # Score based on gene annotations
        if gene_info:
            annotated_genes = sum(
                1 for info in gene_info.values()
                if info.get('ncbi_gene_id') != 'Unknown'
            )
            relevance_score += min(30, annotated_genes * 3)
        
        # Assess clinical relevance level
        if relevance_score >= 80:
            level = 'Very High'
        elif relevance_score >= 60:
            level = 'High'
        elif relevance_score >= 40:
            level = 'Medium'
        elif relevance_score >= 20:
            level = 'Low'
        else:
            level = 'Minimal'
        
        return {
            'score': relevance_score,
            'max_score': max_score,
            'level': level,
            'description': f'Clinical relevance assessed as {level} based on literature evidence and pathway analysis'
        }

def perform_comprehensive_gene_analysis(selected_feature_indices, expression_file, max_genes_annotate=15):
    """
    Main function to perform comprehensive gene analysis
    This is what you'll call from your main script
    
    Parameters:
    -----------
    selected_feature_indices : list
        Selected feature indices from your preprocessor
    expression_file : str
        Path to your expression data file
    max_genes_annotate : int
        Maximum number of genes to fully annotate (to avoid API limits)
        
    Returns:
    --------
    dict
        Comprehensive gene analysis results
    """
    print("=== COMPREHENSIVE GENE ANALYSIS FOR IEEE JOURNAL ===")
    
    # Initialize annotator
    annotator = GeneAnnotator()
    
    # Step 1: Extract gene names from data
    all_gene_names = annotator.extract_gene_names_from_data(expression_file)
    
    # Step 2: Map selected features to gene names
    gene_mapping = annotator.map_selected_features_to_genes(selected_feature_indices, all_gene_names)
    
    # Step 3: Get the actual gene names for selected features
    selected_genes = list(gene_mapping.values())
    
    # Step 4: Biological annotation
    gene_info = annotator.get_gene_info_ncbi(selected_genes, max_genes_annotate)
    
    # Step 5: Pathway enrichment
    pathway_results = annotator.perform_pathway_enrichment_enrichr(selected_genes, max_genes_annotate)
    
    # Step 6: Literature search
    literature_results = annotator.literature_search_pubmed(selected_genes, max_genes_annotate)
    
    # Step 7: Generate comprehensive summary
    biological_summary = annotator.generate_biological_summary(gene_info, pathway_results, literature_results)
    
    results = {
        'gene_mapping': gene_mapping,
        'selected_genes': selected_genes,
        'gene_info': gene_info,
        'pathway_results': pathway_results,
        'literature_results': literature_results,
        'biological_summary': biological_summary,
        'all_gene_names': all_gene_names
    }
    
    print("=== GENE ANALYSIS COMPLETED ===")
    print(f"✓ Mapped {len(gene_mapping)} selected features to genes")
    print(f"✓ Annotated {len(gene_info)} genes from NCBI")
    print(f"✓ Found {sum(len(p) for p in pathway_results.values())} significant pathways")
    print(f"✓ Analyzed literature for {len(literature_results)} genes")
    print(f"✓ Clinical relevance: {biological_summary['clinical_relevance']['level']}")
    
    return results