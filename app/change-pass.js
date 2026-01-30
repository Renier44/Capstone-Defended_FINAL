import React, { useCallback, useState } from 'react';
import { 
    View, 
    Text, 
    StyleSheet, 
    SafeAreaView, 
    TouchableOpacity, 
    TextInput,
    Alert,
    ActivityIndicator,
    ScrollView
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';

// --- MOCK AUTHENTICATION MODULE ---
// In a real application, this logic would live on your backend server 
// and interact with a database (like Firestore, as recommended). 
// Here, we simulate the persistent user credential for testing the change flow.
let __mockUserAuth = {
    // This is the initial "current" password. 
    // You can pretend this is the password you used to "log in."
    password: 'password123', 
};

// Function to simulate a successful password update on the "server"
const updateMockPassword = (newPassword) => {
    __mockUserAuth.password = newPassword;
    console.log(`MOCK AUTH: Password successfully updated to "${newPassword}"`);
};

// Function to check if a password matches the mock stored password
const checkMockPassword = (password) => {
    return password === __mockUserAuth.password;
};
// --- END MOCK AUTHENTICATION MODULE ---


export default function ChangePassword() {
    const [currentPassword, setCurrentPassword] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [message, setMessage] = useState('');

    const handleBack = useCallback(() => {
        console.log('ACTION: Go back to Settings screen');
        // In a real app, you would use router.back() here
    }, []);

    const handleSave = async () => {
        setMessage(''); // Clear previous error messages

        // 1. Client-Side Validation
        if (!currentPassword || !newPassword || !confirmPassword) {
            setMessage('Please fill in all fields.');
            return;
        }

        if (newPassword !== confirmPassword) {
            setMessage('New passwords do not match.');
            return;
        }
        
        if (newPassword.length < 6) {
             setMessage('New password must be at least 6 characters long.');
            return;
        }
        
        // Prevent changing to the same password
        if (newPassword === currentPassword) {
            setMessage('New password cannot be the same as the current password.');
            return;
        }


        // 2. Mock Backend/Auth Check
        if (!checkMockPassword(currentPassword)) {
            setMessage('The current password you entered is incorrect.');
            // Simulate a brief delay to match API response time
            await new Promise(resolve => setTimeout(resolve, 500)); 
            return;
        }


        // 3. Simulated API call to change password
        setIsLoading(true);
        await new Promise(resolve => setTimeout(resolve, 1500)); 
        
        // After simulated API call, update the mock backend and show success
        updateMockPassword(newPassword); 

        setIsLoading(false);
        
        // Show success alert and reset form
        Alert.alert(
            "Success!", 
            `Your password has been successfully updated. The new mock password is: "${newPassword}". Try to "log in" with it!`,
            [{ text: "OK", onPress: () => {
                // Reset form fields on success
                setCurrentPassword('');
                setNewPassword('');
                setConfirmPassword('');
                setMessage(''); // Ensure message is cleared
            }}]
        );
    };
    
    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={handleBack}>
                    <MaterialIcons name="arrow-back-ios" size={24} color="#005A9C" />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Change Password</Text>
                <View style={{ width: 24 }} />
            </View>
            <ScrollView contentContainerStyle={styles.scrollContent}>
                <View style={styles.formContainer}>
                    <Text style={styles.securityNote}>
                        Your current mock password is: **{__mockUserAuth.password}**. 
                        Use this to test the functionality below.
                    </Text>
                    
                    <Text style={styles.inputLabel}>Current Password</Text>
                    <TextInput
                        style={styles.input}
                        value={currentPassword}
                        onChangeText={setCurrentPassword}
                        secureTextEntry
                        placeholder="Enter current password"
                        placeholderTextColor="#999"
                    />

                    <Text style={styles.inputLabel}>New Password</Text>
                    <TextInput
                        style={styles.input}
                        value={newPassword}
                        onChangeText={setNewPassword}
                        secureTextEntry
                        placeholder="Enter new password (min 6 chars)"
                        placeholderTextColor="#999"
                    />

                    <Text style={styles.inputLabel}>Confirm New Password</Text>
                    <TextInput
                        style={styles.input}
                        value={confirmPassword}
                        onChangeText={setConfirmPassword}
                        secureTextEntry
                        placeholder="Re-enter new password"
                        placeholderTextColor="#999"
                    />

                    {message ? (
                        <Text style={styles.errorMessage}>{message}</Text>
                    ) : null}

                    <TouchableOpacity 
                        style={[styles.saveButton, isLoading && styles.saveButtonDisabled]}
                        onPress={handleSave}
                        disabled={isLoading}
                    >
                        {isLoading ? (
                            <ActivityIndicator color="#FFFFFF" />
                        ) : (
                            <Text style={styles.saveButtonText}>Save Changes</Text>
                        )}
                    </TouchableOpacity>

                    
                </View>
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
        justifyContent: 'space-between',
        paddingHorizontal: 20,
    },
    headerTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: '#005A9C'
    },
    scrollContent: {
        paddingVertical: 20,
        flexGrow: 1,
    },
    formContainer: {
        marginHorizontal: 20,
        marginBottom: 20,
        backgroundColor: '#C8EAF7',
        borderRadius: 25,
        padding: 30,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.1,
        shadowRadius: 10,
        elevation: 10,
        flex: 1,
    },
    inputLabel: {
        fontSize: 16,
        fontWeight: '600',
        color: '#005A9C',
        marginBottom: 8,
        marginTop: 15,
    },
    input: {
        backgroundColor: '#FFFFFF',
        borderRadius: 12,
        paddingHorizontal: 15,
        paddingVertical: 12,
        fontSize: 16,
        borderWidth: 1,
        borderColor: '#B0DAE6',
        color: '#333',
    },
    saveButton: {
        backgroundColor: '#005A9C',
        borderRadius: 15,
        padding: 15,
        alignItems: 'center',
        marginTop: 30,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 5,
        elevation: 8,
    },
    saveButtonDisabled: {
        backgroundColor: '#A0BCC7',
    },
    saveButtonText: {
        color: '#FFFFFF',
        fontSize: 18,
        fontWeight: '700',
    },
    errorMessage: {
        color: '#E53935',
        textAlign: 'center',
        marginTop: 20,
        fontSize: 14,
        fontWeight: '500',
    },
    securityNote: {
        color: '#555',
        fontSize: 12,
        textAlign: 'center',
        marginTop: 20,
        backgroundColor: '#F7F7C8',
        padding: 10,
        borderRadius: 10,
        borderWidth: 1,
        borderColor: '#E6DAA0',
    }
});
