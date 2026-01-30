import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    SafeAreaView,
    ScrollView,
    TextInput,
    Image,
    Alert,
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import * as SecureStore from 'expo-secure-store';

// NOTE: ngrok URLs change frequently. Ensure this URL is current and active.
const API_BASE_URL = 'https://2b7bf55b1e09.ngrok-free.app/api';

export default function EditProfile() {
    const router = useRouter();

    // State for the form inputs, initialized to empty strings to prevent render errors
    const [firstName, setFirstName] = useState('');
    const [lastName, setLastName] = useState('');
    const [email, setEmail] = useState('');
    const [dateOfBirth, setDateOfBirth] = useState('DD / MM / YYYY');
    const [profileImage, setProfileImage] = useState(null);
    const [isSaving, setIsSaving] = useState(false);
    const [fullNameDisplay, setFullNameDisplay] = useState('Loading...');

    // Helper to load user data from SecureStore
    const loadLocalUserProfile = async () => {
        try {
            const localProfileStr = await SecureStore.getItemAsync('userProfile');
            if (!localProfileStr) {
                console.warn('No local profile found. Redirecting to login.');
                router.replace('/login');
                return;
            }

            const localProfile = JSON.parse(localProfileStr);

            let apiFirstName = localProfile.first_name || '';
            let apiLastName = localProfile.last_name || '';
            const userEmail = localProfile.email || '';

            let first = apiFirstName;
            let last = apiLastName;

            if (first && !last && first.includes(' ')) {
                const nameParts = first.trim().split(' ');
                first = nameParts[0] || '';
                last = nameParts.slice(1).join(' ') || '';
            }

            setFirstName(first);
            setLastName(last);
            setEmail(userEmail);

            const display = `${first} ${last}`.trim() || localProfile.name || 'Guest User';
            setFullNameDisplay(display);

            setProfileImage(localProfile.profile_image || null);

        } catch (error) {
            console.error('Failed to load local user profile or JSON parse error:', error);
            Alert.alert('Error', 'Could not load profile data. Please log in again.');
            setFullNameDisplay('Error Loading Profile');
            router.replace('/login');
        }
    };

    useEffect(() => {
        loadLocalUserProfile();
    }, []);

    const handleUpdateProfile = async () => {
        if (isSaving) return;

        if (!firstName || !lastName || !email) {
            Alert.alert('Missing Fields', 'Please ensure First Name, Last Name, and Email are filled.');
            return;
        }

        setIsSaving(true);
        const userToken = await SecureStore.getItemAsync('userToken');

        if (!userToken) {
            Alert.alert('Authentication Error', 'You are not logged in.');
            router.replace('/login');
            return;
        }

        try {
            // Reverting to plain JSON for text-only updates
            const payload = {
                first_name: firstName,
                last_name: lastName,
                email: email,
            };

            const response = await fetch(`${API_BASE_URL}/update-profile/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json', // Re-enabling JSON content type
                    'Authorization': `Token ${userToken}`,
                },
                body: JSON.stringify(payload), // Sending plain JSON data
            });

            // --- Enhanced Error Handling ---
            if (!response.ok) {
                let errorDetails = `Status ${response.status}: Failed to update profile.`;
                let serverResponse = '';

                try {
                    const errorData = await response.clone().json();
                    serverResponse = JSON.stringify(errorData, null, 2);

                    if (response.status === 400 && errorData) {
                        const fieldErrors = Object.values(errorData).flat().join('\n');
                        errorDetails = `Validation Error(s):\n${fieldErrors}`;
                    } else if (response.status === 401 || response.status === 403) {
                        errorDetails = 'Authentication failed. Please check your token or log in again.';
                        router.replace('/login');
                    } else {
                        errorDetails = `Server Error (${response.status}): See console for details.`;
                    }

                } catch (e) {
                    serverResponse = await response.clone().text();
                    errorDetails = `Server Error (${response.status}): Server returned non-JSON data. Response: "${serverResponse.substring(0, 100)}..."`;
                }

                console.error('API Update Error:', errorDetails, '\nServer Response:', serverResponse);
                Alert.alert('Update Failed', errorDetails);
                return;
            }
            // --- End Enhanced Error Handling ---


            // If response.ok is true (e.g., 200/201), continue processing success
            const data = await response.json();

            const existingProfileStr = await SecureStore.getItemAsync('userProfile');
            const existingProfile = JSON.parse(existingProfileStr || '{}');

            const newFullName = `${firstName} ${lastName}`;

            const updatedProfile = {
                ...existingProfile,
                first_name: firstName,
                last_name: lastName,
                email: email,
                name: newFullName,
                ...data, // Merge any fields returned by the server
            };

            await SecureStore.setItemAsync('userProfile', JSON.stringify(updatedProfile));

            // Reload the local profile data to ensure state is synchronized with SecureStore
            await loadLocalUserProfile();

            Alert.alert('Success', 'Profile updated successfully!');

            router.back();

        } catch (error) {
            console.error('Update profile network/fetch error:', error);
            Alert.alert(
                'Connection Error',
                'Could not connect to the API. Ensure your device has internet and the API_BASE_URL is correct and active.'
            );
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
                    <MaterialIcons name="arrow-back-ios" size={24} color="#005A9C" />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Edit Profile</Text>
                <View style={{ width: 44 }} />
            </View>
            <ScrollView contentContainerStyle={styles.scrollContent}>
                <View style={styles.profileContainer}>
                    <View style={styles.imageContainer}>
                        <Image
                            source={{
                                uri:
                                    profileImage && profileImage !== 'null'
                                        ? profileImage
                                        : 'https://cdn-icons-png.flaticon.com/512/706/706830.png',
                            }}
                            style={styles.profilePic}
                        />
                        {/* Edit icon removed */}
                    </View>
                    <Text style={styles.userName}>{fullNameDisplay}</Text>
                </View>

                <View style={styles.formContainer}>
                    <View style={styles.inputGroup}>
                        <Text style={styles.label}>First Name</Text>
                        <TextInput
                            style={styles.input}
                            value={firstName}
                            onChangeText={setFirstName}
                            placeholder="First Name"
                        />
                    </View>

                    <View style={styles.inputGroup}>
                        <Text style={styles.label}>Last Name</Text>
                        <TextInput
                            style={styles.input}
                            value={lastName}
                            onChangeText={setLastName}
                            placeholder="Last Name"
                        />
                    </View>

                    <View style={styles.inputGroup}>
                        <Text style={styles.label}>Email</Text>
                        <TextInput
                            style={styles.input}
                            value={email}
                            onChangeText={setEmail}
                            placeholder="Email"
                            keyboardType="email-address"
                            autoCapitalize="none"
                        />
                    </View>

                    <View style={styles.inputGroup}>
                        <Text style={styles.label}>Date Of Birth</Text>
                        <TextInput
                            style={styles.input}
                            value={dateOfBirth}
                            onChangeText={setDateOfBirth}
                            placeholder="DD / MM / YYYY"
                            editable={false}
                        />
                    </View>
                </View>

                <TouchableOpacity
                    style={[styles.updateButton, isSaving && { opacity: 0.7 }]}
                    onPress={handleUpdateProfile}
                    disabled={isSaving}
                >
                    <Text style={styles.updateButtonText}>
                        {isSaving ? 'Updating...' : 'Update Profile'}
                    </Text>
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
    header: {
        height: 56,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'flex-start',
        paddingHorizontal: 20,
    },
    backButton: {
        width: 44,
        paddingVertical: 10,
    },
    headerTitle: {
        flex: 1,
        textAlign: 'center',
        fontSize: 20,
        fontWeight: '700',
        color: '#005A9C'
    },
    scrollContent: {
        paddingVertical: 20,
        paddingHorizontal: 20,
    },
    profileContainer: {
        alignItems: 'center',
        marginBottom: 30,
    },
    imageContainer: {
        position: 'relative',
        marginBottom: 10,
    },
    profilePic: {
        width: 100,
        height: 100,
        borderRadius: 50,
        borderWidth: 3,
        borderColor: '#C8EAF7',
    },
    // The editIcon style definition was removed here.

    // Form Styles
    formContainer: {
        marginBottom: 30,
    },
    inputGroup: {
        marginBottom: 15,
    },
    label: {
        fontSize: 16,
        color: '#005A9C',
        fontWeight: '500',
        marginBottom: 5,
    },
    input: {
        backgroundColor: '#C8EAF7',
        borderRadius: 15,
        height: 50,
        paddingHorizontal: 15,
        fontSize: 16,
        color: '#333',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 3,
        elevation: 3,
    },

    // Update Button
    updateButton: {
        backgroundColor: '#4C6CD5',
        borderRadius: 25,
        height: 55,
        justifyContent: 'center',
        alignItems: 'center',
        marginHorizontal: 10,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 5,
        elevation: 5,
    },
    updateButtonText: {
        color: '#fff',
        fontWeight: 'bold',
        fontSize: 18,
    },
});
