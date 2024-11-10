import React from 'react';

interface BaseAlertProps {
  variant?: 'default' | 'destructive';
  className?: string;
}

interface SimpleAlertProps extends BaseAlertProps {
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  children?: never;
}

interface ComposedAlertProps extends BaseAlertProps {
  children: React.ReactNode;
  message?: never;
  type?: never;
}

type AlertProps = SimpleAlertProps | ComposedAlertProps;

interface AlertDescriptionProps {
  children: React.ReactNode;
  className?: string;
}

export const Alert: React.FC<AlertProps> = (props) => {
  const { className = '' } = props;
  
  const getVariantStyles = (variant: string = 'default') => {
    switch (variant) {
      case 'destructive':
        return 'bg-red-900/20 text-red-200 border-red-500/50';
      default:
        return 'border-border';
    }
  };

  if ('message' in props && props.type) {
    const getTypeStyles = (type: 'info' | 'success' | 'warning' | 'error') => {
      switch (type) {
        case 'info':
          return 'bg-blue-100 text-blue-800 border-l-4 border-blue-500';
        case 'success':
          return 'bg-green-100 text-green-800 border-l-4 border-green-500';
        case 'warning':
          return 'bg-yellow-100 text-yellow-800 border-l-4 border-yellow-500';
        case 'error':
          return 'bg-red-100 text-red-800 border-l-4 border-red-500';
        default:
          return 'bg-gray-100 text-gray-800 border-l-4 border-gray-500';
      }
    };

    return (
      <div 
        className={`p-4 rounded ${getTypeStyles(props.type)} ${className}`}
        role="alert"
      >
        {props.message}
      </div>
    );
  } else {
    return (
      <div
        className={`relative w-full rounded-lg border p-4 ${getVariantStyles(props.variant)} ${className}`}
        role="alert"
      >
        {props.children}
      </div>
    );
  }
};

export const AlertDescription: React.FC<AlertDescriptionProps> = ({
  children,
  className = '',
}) => {
  return (
    <div className={`text-sm [&_p]:leading-relaxed ${className}`}>
      {children}
    </div>
  );
};