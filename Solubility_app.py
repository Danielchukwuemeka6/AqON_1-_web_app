# import the necessary libraries
import streamlit as st



from distutils.command.upload import upload
import numpy as np
import pandas as pd 
from matplotlib import image, pyplot as plt
import matplotlib.patches as mpatches 
import seaborn as sns
# Sklearn --------------------------------------------------------------
from sklearn.metrics import r2_score, mean_squared_error
import pickle 
# RDKIt----------------------------------------------------------------
from rdkit.Chem import AllChem 
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors 
from rdkit.ML.Descriptors import MoleculeDescriptors

# streamlit ----------------------------------------------------------

from PIL import Image
import base64
import io

#=======================================================================================================
#  Use trained lgbm regressor and standard scaler for predicting aqueous solubility of organic compounds
#=======================================================================================================



with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
   
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)
#--------------------------------------------------------------------------------------------------------------
# List of molecular discriptors used in the training set. These descriptors should also be used for the test set
#---------------------------------------------------------------------------------------------------------------
descriptor_columns = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea'] 

# Adding Radio button
option = st.sidebar.radio('Pages',options= ['Home','Developer_ID'])

if option == 'Home':
    st.title('AqON_1 SOLUBILITY WEB APP') 
    st.sidebar.markdown('<h2 style="color:#5a03fc;background-color:powderblue;border-radius:10px;text-align:center"> Use this Sidebar for Solubility Prediction </h2>',unsafe_allow_html=True)


    # Display my profile
    st.markdown(""" This Web Application was developed by [Ngwu Comfort pHD & Ottah Daniel Chukwuemeka,BSc.] for Aqeous Solubility Prediction of Organic molecules""")

    # Define solubility and explain why it is important
    st.markdown('`Solubility`, a crucial concept, is defined as the maximum quantity of solute capable of dissolving in a specific volume of solvent to produce a saturated solution at a designated temperature—typically room temperature. A fundamental guiding principle is "like dissolves like," signifying that molecules sharing similar polarities and intermolecular forces exhibit a greater tendency to dissolve in one another.The significance of water solubility extends into the realm of drug discovery, directly impacting bioavailability.')

    # A plotting function
    def plot_data(actual, predicted, title):
        # model performance using RMSE
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        # R^2 (coefficient of determination) :
        R2 = r2_score(actual, predicted)
        
        plt.figure(figsize=(8, 6))

        # Plot the figure
        sns.regplot(x=predicted, y=actual, line_kws={"lw": 2, 'ls': '--', 'color': 'red', "alpha": 0.7})
        plt.title(title, color='red')
        plt.xlabel('Predicted logS(mol/L)', color='blue')
        plt.ylabel('Experimental logS(mol/L)', color='blue')
        plt.xlim(-8, 1)

        plt.grid(alpha=0.3)
        R2_patch = mpatches.Patch(label="R2={:04.2f}".format(R2))
        rmse_patch = mpatches.Patch(label="RMSE={:04.2f}".format(rmse))
        plt.legend(handles=[R2_patch, rmse_patch])

        # Display the plot using Streamlit
        st.pyplot(plt)

    # Read test data
    test = pd.read_csv('results.csv')

    # Assuming you have columns named 'Actual' and 'Predicted'
    actual_values = test['Actual']
    predicted_values = test['Predicted']

    # Call the plotting function
    plot_data(actual_values, predicted_values, title='Actual vs Predicted Solubility')

    # Calculate the 200 RDKit Descriptors ==============================================
    def RDkit_descriptors(smiles):
        mols = [Chem.MolFromSmiles(i) for i in smiles] 
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] 
                                        for x in Descriptors._descList])
        desc_names = calc.GetDescriptorNames()
        Mol_descriptors =[]
        for mol in mols:
            # add hydrogens to molecules
            mol=Chem.AddHs(mol)
            # Calculate all available descriptors for each molecule
            descriptors = calc.CalcDescriptors(mol)
            Mol_descriptors.append(descriptors)
        return Mol_descriptors,desc_names 

    #--------- A function that can generate a csv file for output file to download
    # Big credit : https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/2
    #           https://github.com/dataprofessor/ml-auto-app/blob/main/app.py
    def file_download(data,file):
        df = data.to_csv(index=False)
        f= base64.b64encode(df.encode()).decode()
        link = f'<a href ="data:file/csv; base64,{f}" download={file}> Download {file} file</a>'
        return link

    #  User input--------------------------------------------------------------------------------------------------------------
    # 1. One or few SMILES input
    one_or_few_SMILES = st.sidebar.text_input('Enter SMILE Strings in single or double quotation separated by comma:',"['CCCCO']")
    st.sidebar.markdown('''`or upload SMILE strings in CSV format, note that SMILE strings of the molecules should be in 'SMILES' column:`''')

    # 2.  upload many SMILES input
    many_SMILES = st.sidebar.file_uploader("=====================================")
    # ---------------------------------------------------------------------------------------------------------------------------
    st.sidebar.markdown("""**If you upload your CSV file, click the button below to get the solubility prediction** """)
    prediction = st.sidebar.button('Predict logS of molecules')

    # Define the classification function
    def classify_solubility(value):
        if value < -4:
            return "INSOLUBLE"
        elif value > 0:
            return "HIGHLY_SOLUBLE"
        elif -4 <= value <= -2:
            return "LOW_SOLUBILITY"
        elif -2 < value <= 0:
            return "MODERATELY_SOLUBLE"
        else:
            return ""

    if one_or_few_SMILES != "['CCCCO']":
        df = pd.DataFrame(eval(one_or_few_SMILES), columns =['SMILES'])
        #========= function call to calculate 200 molecular descriptors using SMILES
        Mol_descriptors,desc_names = RDkit_descriptors(df['SMILES'])
        #========= Put the 200 molecular descriptors in  data frame
        test_set_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)
        #========= Use only the 200 descriptors listed above
        X_test = test_set_with_200_descriptors[descriptor_columns]

        #======== The data was standardized during traning and test set also need to be standardized
        X_test_scaled = scaler.transform(X_test)
        #---------------------------------------------------------------------

        #======== Prediction of solubility using model1(LightGBM) and model2(HistGradientBoostingRegressor)

        

        X_logS = model.predict(X_test_scaled) 

        #======= Put the predicted solubility in Dataframe
        predicted = pd.DataFrame(X_logS, columns =['Predicted logS (mol/L)']) 
        # Apply the classification function to create the "class" column
        predicted['class'] = predicted['Predicted logS (mol/L)'].apply(classify_solubility)

        #======= Concatenate SMILES and the predicted solubility 
        output = pd.concat([df,predicted], axis=1)
        st.sidebar.markdown('''## See your output in the following table:''')

        #======= Display output in table form
        st.sidebar.write(output)

        #======= show CSV file attachment
        st.sidebar.markdown(file_download(output,"predicted_logS.csv"),unsafe_allow_html=True)

    #===== Use uploaded SMILES to calculate their logS values
    elif prediction:
        df2 = pd.read_csv(many_SMILES)
        Mol_descriptors, desc_names = RDkit_descriptors(df2['SMILES'])
        test_set_with_200_descriptors = pd.DataFrame(Mol_descriptors, columns=desc_names)
        X_test = test_set_with_200_descriptors[descriptor_columns]
        # transform the test data
        X_test_scaled = scaler.transform(X_test)
        # predict solubility
        X_logS = model.predict(X_test_scaled)
        # put the predicted values in dataframe
        predicted = pd.DataFrame(X_logS, columns=['Predicted logS (mol/L)'])
        # Apply the classification function to create the "class" column
        predicted['class'] = predicted['Predicted logS (mol/L)'].apply(classify_solubility)
        # concatenate SMILES and predicted solubility values on a dataframe
        output = pd.concat([df2['SMILES'], predicted], axis=1)
        st.sidebar.markdown('''## Your output is shown in the following table:''')
        # display the output on the sidebar
        st.sidebar.write(output)
        # display a download csv file as a link on the sidebar
        st.sidebar.markdown(file_download(output, "predicted_logS.csv"), unsafe_allow_html=True)
    else:
        st.markdown('<div style="border: 2px solid #4908d4;border-radius:20px;padding: 3%;text-align:center"><h5> To use this model,  please use the sidebar. For few molecules, you can directly put the SMILES in a single or double quotations separated by comma in the sidebar. For many molecules, you can put their SMILES strings in a "SMILES" column, upload them and click the button which says "Predict logS of moleclues" shown in the sidebar.</h5> <h5 style="color:white;background-color:#0a0605;border-radius:10px;padding: 3%;opacity: 0.7;">Please also note that predcition is more reliable if the compounds to be predicted are similar with training dataset that is logS values ranges between -8.8 and 2.2.</h5></div>',unsafe_allow_html=True)

elif option == 'Developer_ID':
    st.title("DEVELOPER'S ID")
    st.markdown('This web app was built by Dr. Ngwu Comfort PhD and her project student Ottah Daniel Chukwuemeka Bsc.')

    from PIL import Image
    import io

    def main():

        # Provide the local file path
        image_path = r"C:\Users\daniel\Pictures\1656420086629.jpeg"

        # Open the image using PIL
        image = Image.open(image_path)

        # Convert the image to bytes
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')

        # Display the image using st.image
        st.image(image_bytes, caption="Author 1", use_column_width=True)
        st.markdown('''
                    ## Ngwu Comfort, PhD (Environmental Chemistry).
                    ### Michael Okpara University of Agriculture, Umudike
                    ### Chemistry Department
                    ### kom4tngwu@gmail.com
                    ### +234-803-214-5757
                    
                    ''')


    if __name__ == "__main__":
        main()

    def main():
        
        # Provide the local file path
        image_path = r"C:\Users\daniel\Pictures\portfolio_picture.jpg"

        # Open the image using PIL
        image = Image.open(image_path)

        # Convert the image to bytes
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')

        # Display the image using st.image
        st.image(image_bytes, caption="Author 2", use_column_width=True)
        st.markdown('''
                    ## Ottah Daniel Chukwuemeka, BSc.
                    ## Cheminformatics-Data-Scientist
                    ### Michael Okpara University of Agriculture, Umudike
                    ### Chemistry Department
                    ### danielottah6@gmail.com
                    ### +234-704-508-5599
                    
                    ''')



    if __name__ == "__main__":
        main()


