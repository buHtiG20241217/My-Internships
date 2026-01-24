class ExplanationGenerator:
    def __init__(self):
        self.price_map = {'low': 1, 'medium': 2, 'high': 3, 'premium': 4}

    def generate_explanation(self, user_input, service_row):
        """
        Generates a list of reasons why a service was recommended.
        user_input: Dict of user preferences.
        service_row: Dict or Series of service attributes.
        """
        reasons = []
        
        # 1. Business Type Match
        user_type = user_input.get('Target_Business_Type', '').lower()
        svc_type = str(service_row.get('Target_Business_Type', '')).lower()
        if user_type and user_type == svc_type:
            reasons.append(f"Perfect match for {svc_type.title()} businesses.")
        elif user_type:
            # Maybe it's a generic service?
            pass

        # 2. Price Match
        user_price = user_input.get('Price_Category', 'medium').lower()
        svc_price = str(service_row.get('Price_Category', '')).lower()
        
        u_p_val = self.price_map.get(user_price, 2)
        s_p_val = self.price_map.get(svc_price, 2)
        
        if s_p_val <= u_p_val:
            reasons.append("Within your budget.")
        else:
            reasons.append(f"Slightly above budget ({svc_price.title()}).")

        # 3. Location Match
        user_loc = user_input.get('Location_Area', '').lower()
        svc_loc = str(service_row.get('Location_Area', '')).lower() # Use column name from enc
        # Note: In the dict passed from engine, it might be 'Location' or 'Location_Area' depending on how I constructed 'results' in engine.
        # Let's check RecommendationEngine.get_recommendations construction.
        # It keys it as 'Location'.
        if not svc_loc:
             svc_loc = str(service_row.get('Location', '')).lower()

        if svc_loc == 'remote':
            reasons.append("Available remotely.")
        elif user_loc and user_loc == svc_loc:
            reasons.append(f"Located in {svc_loc.title()}.")
            
        # 4. Language Match
        user_langs = user_input.get('Language_Support', [])
        if isinstance(user_langs, str): user_langs = [user_langs]
        user_langs = [l.lower() for l in user_langs]
        
        svc_lang = str(service_row.get('Language_Support', '')).lower()
        
        # Logic: If service is 'both', it covers hindi/english.
        # If service is 'regional', distinct.
        matched_lang = False
        if 'both' in svc_lang:
            if 'english' in user_langs or 'hindi' in user_langs:
                matched_lang = True
        elif svc_lang in user_langs:
            matched_lang = True
            
        if matched_lang:
            reasons.append("Supports your preferred language.")

        # 5. Quality Badge
        # Assuming 'Match_Quality' or implicit high score
        # We can add a generic one if score is very high (passed in?)
        # For now, let's keep it based on attributes.

        if not reasons:
            reasons.append("Matches your description.")

        return reasons

if __name__ == "__main__":
    gen = ExplanationGenerator()
    
    # Mock Data
    u_input = {
        'Target_Business_Type': 'Restaurant',
        'Price_Category': 'Low',
        'Location_Area': 'Delhi',
        'Language_Support': ['Hindi']
    }
    
    svc_row_1 = {
        'Target_Business_Type': 'Restaurant',
        'Price_Category': 'Low',
        'Location_Area': 'Remote',
        'Language_Support': 'Hindi'
    }
    
    print("User:", u_input)
    print("Service:", svc_row_1)
    print("Explanation:", gen.generate_explanation(u_input, svc_row_1))
