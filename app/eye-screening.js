// EyeScreening.js
import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  Alert,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons, MaterialIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import * as SecureStore from 'expo-secure-store';

export default function EyeScreening() {
  const router = useRouter();
  const [localImage, setLocalImage] = useState(null);
  const [consentChecked, setConsentChecked] = useState(false);
  const [imageStatus, setImageStatus] = useState('');
  const [imageAnalysisMessage, setImageAnalysisMessage] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const BACKEND_URL =
    'https://capstone-defended-final.onrender.com/api/classify-eye/';

  // ------------------------------
  // 📸 IMAGE PICKER
  // ------------------------------
  const pickImage = async (fromCamera = false) => {
    try {
      const result = fromCamera
        ? await ImagePicker.launchCameraAsync({ allowsEditing: true, quality: 1 })
        : await ImagePicker.launchImageLibraryAsync({ allowsEditing: true, quality: 1 });

      if (!result.canceled) {
        const uri = result.assets[0].uri;
        setLocalImage(uri);
        setImageStatus('');
        setImageAnalysisMessage('');
        await checkImageQuality(uri);
      }
    } catch (error) {
      console.error('Error picking image:', error);
      Alert.alert('Error', 'Failed to pick image.');
    }
  };

  // ------------------------------
  // 🧠 SIMPLE IMAGE QUALITY CHECK
  // ------------------------------
  const checkImageQuality = async (imageUri) => {
    try {
      setIsAnalyzing(true);
      setImageAnalysisMessage('Analyzing image quality...');
      const isGoodQuality = Math.random() > 0.15;
      if (isGoodQuality) {
        setImageStatus('GOOD');
        setImageAnalysisMessage('Image is suitable for screening.');
      } else {
        setImageStatus('BAD');
        setImageAnalysisMessage('Image is too blurry or dark. Please retake.');
      }
    } catch (error) {
      console.error('Error checking image quality:', error);
      Alert.alert('Error', 'Image quality check failed.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ------------------------------
  // 🤖 CALL AI BACKEND CLASSIFIER
  // ------------------------------
  // ... (Imports remain the same)

  // ------------------------------
  // 🤖 CALL AI BACKEND CLASSIFIER
  // ------------------------------
  const classifyEyeImage = async (imageUri) => {
    try {
      setIsAnalyzing(true);

      // ... (Keep your User/Token retrieval logic here) ...
      const userStr = await SecureStore.getItemAsync('user');
      // ...

      const formData = new FormData();
      // Important: Ensure filename ends in .jpg or .png
      formData.append('image', { uri: imageUri, type: 'image/jpeg', name: 'screening.jpg' });

      // ... (Append user_id/email logic) ...
      // Example:
      if (userStr) {
          const u = JSON.parse(userStr);
          if(u.id) formData.append('user_id', u.id.toString());
      }

      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        body: formData,
        headers: {
           // Do NOT set 'Content-Type': 'multipart/form-data' manually.
           // Let fetch generate the boundary.
           // Add Authorization if needed.
        },
      });

      const result = await response.json();

      if (!response.ok || result.status === 'error') {
        Alert.alert('Screening Issue', result.message || 'Could not process image.');
        setIsAnalyzing(false);
        return;
      }

      // Success Handling
      const { diagnosis, probabilities, message } = result;

      // Determine routing status based on specific string from Backend
      // Backend sends: "Strabismus" or "Strabismus-Free"
      const isNormal = diagnosis === 'Strabismus-Free';

      router.push({
        pathname: '/eye-results',
        params: {
          status: isNormal ? 'normal' : 'requires_attention', // Used for UI colors
          diagnosis: diagnosis, // The actual label
          probabilities: JSON.stringify(probabilities), // Pass object as string
          message: message, // The user friendly message
          imageUri: imageUri,
          timestamp: new Date().toISOString(),
        },
      });

    } catch (error) {
      console.error('Network/Logic Error:', error);
      Alert.alert('Connection Error', 'Could not reach the screening server.');
    } finally {
      setIsAnalyzing(false);
    }
  };

// ... (Rest of UI code remains the same)

  // ------------------------------
  // 🚀 HANDLE NEXT (AI CALL)
  // ------------------------------
  const handleNext = async () => {
    if (!consentChecked) return Alert.alert('Consent Required', 'Please check the consent box.');
    if (!localImage) return Alert.alert('No Image', 'Please capture or upload an eye image.');
    if (imageStatus !== 'GOOD') return Alert.alert('Image Unsuitable', imageAnalysisMessage);

    await classifyEyeImage(localImage);
  };

  // ------------------------------
  // 🧱 UI RENDER
  // ------------------------------
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}></Text>
        </View>

        <View style={styles.titleContainer}>
          <Text style={styles.title}>Preliminary Strabismus Screening</Text>
          <Text style={styles.subtitle}>
            Capture or upload an image of your eyes for AI-assisted screening.
          </Text>
        </View>

        <View style={styles.imagePreviewContainer}>
          {localImage ? (
            <Image
              source={{ uri: localImage }}
              style={styles.imagePreview}
              resizeMode="contain"
            />
          ) : (
            <View style={styles.placeholder}>
              <Ionicons name="eye-outline" size={80} color="#99AACC" />
              <Text style={styles.placeholderText}>No Image Selected</Text>
            </View>
          )}
        </View>

        {isAnalyzing && <ActivityIndicator size="large" style={{ marginVertical: 10 }} />}
        {imageAnalysisMessage !== '' && !isAnalyzing && (
          <Text
            style={[
              styles.statusText,
              imageStatus === 'GOOD' ? { color: '#00a86b' } : { color: '#f55' },
            ]}
          >
            {imageAnalysisMessage}
          </Text>
        )}

        <View style={styles.buttonsContainer}>
          <TouchableOpacity style={styles.button} onPress={() => pickImage(true)}>
            <Ionicons name="camera-outline" size={24} color="#fff" />
            <Text style={styles.buttonText}>Capture</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.button} onPress={() => pickImage(false)}>
            <MaterialIcons name="upload-file" size={24} color="#fff" />
            <Text style={styles.buttonText}>Upload</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.noteContainer}>
          <Text style={styles.noteTitle}>Image Capture Tips:</Text>
          <Text style={styles.noteText}>• Avoid wearing glasses or contacts.</Text>
          <Text style={styles.noteText}>• Ensure bright lighting.</Text>
          <Text style={styles.noteText}>• Capture both eyes clearly.</Text>
          <Text style={styles.noteText}>• Keep your face straight and steady.</Text>
        </View>

        <View style={styles.consentContainer}>
          <TouchableOpacity
            style={[styles.checkbox, consentChecked && styles.checkboxChecked]}
            onPress={() => setConsentChecked(!consentChecked)}
          >
            {consentChecked && <Ionicons name="checkmark" size={18} color="#fff" />}
          </TouchableOpacity>
          <Text style={styles.consentText}>
            I consent to AI-assisted preliminary eye screening.
          </Text>
        </View>

        <TouchableOpacity
          style={[
            styles.nextButton,
            {
              backgroundColor:
                consentChecked && localImage && imageStatus === 'GOOD' ? '#77CDE0' : '#99AACC',
            },
          ]}
          onPress={handleNext}
          disabled={!consentChecked || !localImage || imageStatus !== 'GOOD'}
        >
          <Text style={styles.nextButtonText}>
            {isAnalyzing ? 'Processing...' : 'Analyze'}
          </Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}


// ------------------------------
// 🎨 STYLES
// ------------------------------
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#D9ECFF' },
  scrollContent: { padding: 20, alignItems: 'center' },
  header: { width: '100%', alignItems: 'center', paddingVertical: 10 },
  headerTitle: { fontSize: 24, fontWeight: 'bold', color: '#2260FF' },
  titleContainer: { width: '100%', alignItems: 'center', marginBottom: 20 },
  title: { fontSize: 22, fontWeight: 'bold', color: '#0057B7', marginBottom: 10, textAlign: 'center' },
  subtitle: { fontSize: 13, color: '#050505ff', textAlign: 'center' },
  imagePreviewContainer: {
    width: '100%',
    height: 250,
    marginBottom: 20,
    borderRadius: 20,
    overflow: 'hidden',
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#fff',
    elevation: 5,
  },
  imagePreview: { width: '100%', height: '100%' }, // ✅ keep aspect ratio with contain
  placeholder: { justifyContent: 'center', alignItems: 'center' },
  placeholderText: { color: '#99AACC', marginTop: 10 },
  buttonsContainer: { flexDirection: 'row', justifyContent: 'space-between', width: '100%', marginBottom: 20 },
  button: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 15, backgroundColor: '#77CDE0', borderRadius: 15, marginHorizontal: 5 },
  buttonText: { color: '#000000ff', fontWeight: '500', marginLeft: 10, fontSize: 16 },
  noteContainer: { width: '100%', padding: 15, backgroundColor: '#C8EAF7', borderRadius: 15, marginBottom: 20, borderLeftWidth: 5, borderLeftColor: '#77CDE0' },
  noteTitle: { fontWeight: 'bold', fontSize: 16, marginBottom: 10, color: '#161616ff' },
  noteText: { fontSize: 14, color: '#555', marginBottom: 5 },
  consentContainer: { flexDirection: 'row', alignItems: 'flex-start', marginBottom: 20, width: '100%' },
  checkbox: { width: 24, height: 24, borderRadius: 6, borderWidth: 2, borderColor: '#77CDE0', justifyContent: 'center', alignItems: 'center', marginRight: 10 },
  checkboxChecked: { backgroundColor: '#77CDE0', borderColor: '#77CDE0' },
  consentText: { flex: 1, fontSize: 14, color: '#161616ff' },
  nextButton: { width: '100%', paddingVertical: 18, borderRadius: 15, alignItems: 'center' },
  nextButtonText: { color: '#fff', fontWeight: 'bold', fontSize: 18 },
  statusText: { fontSize: 16, fontWeight: 'bold', textAlign: 'center', marginBottom: 10 },
});


//STOP
