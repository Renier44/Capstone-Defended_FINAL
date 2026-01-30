import React, { useState, useContext, useEffect } from 'react';
import {
    View, Text, StyleSheet, TouchableOpacity, Image, Alert, SafeAreaView, ScrollView
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons, MaterialIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { ImageContext } from './context/ImageContext'; // only ImageContext

export default function EyeScreening() {
    const router = useRouter();
    const { setImageUri } = useContext(ImageContext);
    const [localImage, setLocalImage] = useState(null);
    const [consentChecked, setConsentChecked] = useState(false);
    const [imageStatus, setImageStatus] = useState(null); // 'GOOD', 'POOR', or null
    const [imageAnalysisMessage, setImageAnalysisMessage] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    // This function converts a local URI to a Base64 string
    const toBase64 = async (uri) => {
        const response = await fetch(uri);
        const blob = await response.blob();
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                // The result includes "data:image/jpeg;base64,", so we split that off
                resolve(reader.result.split(',')[1]);
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    };

    // This function checks the image quality using Gemini API
    const checkImageQuality = async (imageUri) => {
        setIsAnalyzing(true);
        setImageStatus(null);
        setImageAnalysisMessage(null);
        try {
            const base64ImageData = await toBase64(imageUri);
            
            // The updated prompt for more accurate validation
            const prompt = "Analyze this image for its suitability for a medical eye screening. Assess the focus, lighting, head position, and any obstructions. Provide a concise, single-sentence response. If the image is suitable, say 'Image is suitable for screening.'. If not, explain why, for example: 'Image is blurry and eyes are not centered.' or 'Lighting is poor with glare on the eyes.'";
            
            const apiKey = "";
            const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;

            const payload = {
                contents: [
                    {
                        role: "user",
                        parts: [
                            { text: prompt },
                            {
                                inlineData: {
                                    mimeType: "image/jpeg",
                                    data: base64ImageData
                                }
                            }
                        ]
                    }
                ],
            };

            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();
            const text = result?.candidates?.[0]?.content?.parts?.[0]?.text;
            
            if (text && text.includes('suitable for screening')) {
                setImageStatus('GOOD');
                setImageAnalysisMessage('Image is suitable for screening.');
            } else {
                setImageStatus('POOR');
                setImageAnalysisMessage(text || 'Unable to analyze image.');
            }
        } catch (error) {
            console.error('Error checking image quality:', error);
            setImageStatus('POOR'); // Default to POOR on error
            setImageAnalysisMessage('Analysis failed. Please try again.');
        } finally {
            setIsAnalyzing(false);
        }
    };

    useEffect(() => {
        if (localImage) {
            checkImageQuality(localImage);
        }
    }, [localImage]);

    const handleCaptureImage = async () => {
        const { status } = await ImagePicker.requestCameraPermissionsAsync();
        if (status !== 'granted') {
            Alert.alert('Permission denied', 'Camera access is required.');
            return;
        }

        const result = await ImagePicker.launchCameraAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [4, 3],
            quality: 1,
        });

        if (!result.canceled && result.assets?.length) {
            setLocalImage(result.assets[0].uri);
            setImageUri(result.assets[0].uri);
        }
    };

    const handleUploadImage = async () => {
        const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (status !== 'granted') {
            Alert.alert('Permission denied', 'Gallery access is required.');
            return;
        }

        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [4, 3],
            quality: 1,
        });

        if (!result.canceled && result.assets?.length) {
            setLocalImage(result.assets[0].uri);
            setImageUri(result.assets[0].uri);
        }
    };

    const handleNext = () => {
        if (!consentChecked) {
            Alert.alert('Consent Required', 'Please check the consent box.');
            return;
        }
        if (!localImage) {
            Alert.alert('No Image', 'Please capture or upload an eye image.');
            return;
        }
        if (imageStatus !== 'GOOD') {
            Alert.alert('Image Unsuitable', imageAnalysisMessage || 'The image is not suitable for analysis. Please retake the photo, ensuring proper lighting and angle.');
            return;
        }

        router.push('/check-photo');
    };

    return (
        <SafeAreaView style={styles.container}>
            <ScrollView contentContainerStyle={styles.scrollContent}>
                <View style={styles.header}>
                    <Text style={styles.headerTitle}>Eye Screening</Text>
                </View>
                <View style={styles.titleContainer}>
                    <Text style={styles.title}>Preliminary Strabismus Screening</Text>
                    <Text style={styles.subtitle}>Capture or upload an image of your eyes for preliminary screening.</Text>
                </View>

                <View style={styles.imagePreviewContainer}>
                    {localImage ? (
                        <Image source={{ uri: localImage }} style={styles.imagePreview} resizeMode="cover" />
                    ) : (
                        <View style={styles.placeholder}>
                            <Ionicons name="eye-outline" size={80} color="#99AACC" />
                            <Text style={styles.placeholderText}>No Image Selected</Text>
                        </View>
                    )}
                </View>
                
                {isAnalyzing && (
                    <Text style={styles.statusText}>Analyzing image...</Text>
                )}

                {imageAnalysisMessage && !isAnalyzing && (
                    <Text style={[styles.statusText, imageStatus === 'POOR' ? { color: '#f55' } : { color: '#00a86b' }]}>
                        {imageAnalysisMessage}
                    </Text>
                )}

                <View style={styles.buttonsContainer}>
                    <TouchableOpacity style={styles.button} onPress={handleCaptureImage}>
                        <Ionicons name="camera-outline" size={24} color="#fff" />
                        <Text style={styles.buttonText}>Capture</Text>
                    </TouchableOpacity>

                    <TouchableOpacity style={styles.button} onPress={handleUploadImage}>
                        <MaterialIcons name="upload-file" size={24} color="#fff" />
                        <Text style={styles.buttonText}>Upload</Text>
                    </TouchableOpacity>
                </View>

                <View style={styles.noteContainer}>
                    <Text style={styles.noteTitle}>Important Notes for Image Capture:</Text>
                    <Text style={styles.noteText}>
                        • Avoid wearing eyeglasses or contact lenses.
                    </Text>
                    <Text style={styles.noteText}>
                        • Ensure the image is captured with the right angle and proper lighting.
                    </Text>
                    <Text style={styles.noteText}>
                        • Capture both eyes in the photo, looking straight ahead.
                    </Text>
                    <Text style={styles.noteText}>
                        • Blink normally to keep your eyes moist and prevent redness.
                    </Text>
                    <Text style={styles.noteText}>
                        • Maintain a stable head position and avoid any movement during the capture.
                    </Text>
                </View>

                <View style={styles.consentContainer}>
                    <TouchableOpacity
                        style={[styles.checkbox, consentChecked && styles.checkboxChecked]}
                        onPress={() => setConsentChecked(!consentChecked)}
                    >
                        {consentChecked && <Ionicons name="checkmark" size={18} color="#fff" />}
                    </TouchableOpacity>
                    <Text style={styles.consentText}>
                        I have read the instructions and give consent to use my eye image for screening.
                    </Text>
                </View>

                <TouchableOpacity
                    style={[styles.nextButton, { backgroundColor: (consentChecked && localImage && imageStatus === 'GOOD') ? '#2260FF' : '#99AACC' }]}
                    onPress={handleNext}
                    disabled={!consentChecked || !localImage || imageStatus !== 'GOOD'}
                >
                    <Text style={styles.nextButtonText}>Next</Text>
                </TouchableOpacity>
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#77CDE0'
    },
    scrollContent: {
        padding: 20,
        alignItems: 'center'
    },
    header: {
        width: '100%',
        alignItems: 'center',
        paddingVertical: 10
    },
    headerTitle: {
        fontSize: 24,
        fontWeight: 'bold',
        color: '#2260FF',
        fontFamily: 'LeagueSpartan',
    },
    titleContainer: {
        width: '100%',
        alignItems: 'center',
        marginBottom: 20
    },
    title: {
        fontSize: 22,
        fontWeight: 'bold',
        color: '#FFD54F',
        marginBottom: 10,
        textAlign: 'center',
        fontFamily: 'LeagueSpartan',
    },
    subtitle: {
        fontSize: 16,
        color: '#fff',
        textAlign: 'center',
        fontFamily: 'LeagueSpartan',
    },
    imagePreviewContainer: {
        width: '100%',
        height: 250,
        marginBottom: 20,
        borderRadius: 20,
        overflow: 'hidden',
        backgroundColor: '#C8EAF7',
        justifyContent: 'center',
        alignItems: 'center',
        borderWidth: 2,
        borderColor: '#fff',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.2,
        shadowRadius: 6,
        elevation: 5,
    },
    imagePreview: {
        width: '100%',
        height: '100%'
    },
    placeholder: {
        justifyContent: 'center',
        alignItems: 'center'
    },
    placeholderText: {
        color: '#99AACC',
        marginTop: 10,
        fontFamily: 'LeagueSpartan',
    },
    buttonsContainer: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        width: '100%',
        marginBottom: 20
    },
    button: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 15,
        backgroundColor: '#2260FF',
        borderRadius: 15,
        marginHorizontal: 5,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.15,
        shadowRadius: 3,
        elevation: 3,
    },
    buttonText: {
        color: '#fff',
        fontWeight: '500',
        marginLeft: 10,
        fontSize: 16,
        fontFamily: 'LeagueSpartan',
    },
    noteContainer: {
        width: '100%',
        padding: 15,
        backgroundColor: '#C8EAF7',
        borderRadius: 15,
        marginBottom: 20,
        borderLeftWidth: 5,
        borderLeftColor: '#FFD54F',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 3.84,
        elevation: 5,
    },
    noteTitle: {
        fontWeight: 'bold',
        fontSize: 16,
        marginBottom: 10,
        color: '#2260FF',
        fontFamily: 'LeagueSpartan',
    },
    noteText: {
        fontSize: 14,
        color: '#555',
        marginBottom: 5,
        fontFamily: 'LeagueSpartan',
    },
    consentContainer: {
        flexDirection: 'row',
        alignItems: 'flex-start',
        marginBottom: 20,
        width: '100%'
    },
    checkbox: {
        width: 24,
        height: 24,
        borderRadius: 6,
        borderWidth: 2,
        borderColor: '#2260FF',
        justifyContent: 'center',
        alignItems: 'center',
        marginRight: 10
    },
    checkboxChecked: {
        backgroundColor: '#2260FF',
        borderColor: '#2260FF'
    },
    consentText: {
        flex: 1,
        fontSize: 14,
        color: '#fff',
        fontFamily: 'LeagueSpartan',
    },
    nextButton: {
        width: '100%',
        paddingVertical: 18,
        borderRadius: 15,
        alignItems: 'center',
        shadowColor: '#000',
        shadowOpacity: 0.2,
        shadowOffset: { width: 0, height: 4 },
        shadowRadius: 5,
        elevation: 5,
    },
    nextButtonText: {
        color: '#fff',
        fontWeight: 'bold',
        fontSize: 18,
        fontFamily: 'LeagueSpartan',
    },
    statusText: {
        fontSize: 16,
        fontWeight: 'bold',
        textAlign: 'center',
        marginBottom: 10,
        paddingHorizontal: 10
    }
});
