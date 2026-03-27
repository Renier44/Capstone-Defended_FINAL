import React, { useState, useContext } from 'react';
import {
    View, Text, StyleSheet, TouchableOpacity, Image, Alert, SafeAreaView, ScrollView
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons, MaterialIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { ImageContext } from '../context/ImageContext';

export default function EyeScreening() {
    const router = useRouter();
    const { setImageUri } = useContext(ImageContext);
    const [localImage, setLocalImage] = useState(null);
    const [consentChecked, setConsentChecked] = useState(false);

    const handleCaptureImage = async () => {
        const { status } = await ImagePicker.requestCameraPermissionsAsync();
        if (status !== 'granted') return Alert.alert('Permission denied', 'Camera access is required.');

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
        if (status !== 'granted') return Alert.alert('Permission denied', 'Gallery access is required.');

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
        if (!consentChecked) return Alert.alert('Consent Required', 'Please check the consent box.');
        if (!localImage) return Alert.alert('No Image', 'Please capture or upload an eye image.');
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
                    style={[styles.nextButton, { backgroundColor: consentChecked ? '#2260FF' : '#99AACC' }]}
                    onPress={handleNext}
                    disabled={!consentChecked}
                >
                    <Text style={styles.nextButtonText}>Next</Text>
                </TouchableOpacity>
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    // Note: Custom fonts in React Native require additional steps (like adding the font file to the project)
    // which is not possible in this single-file environment. The 'LeagueSpartan' font family is added
    // here to demonstrate where the change would be applied if the font were available.
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
});
